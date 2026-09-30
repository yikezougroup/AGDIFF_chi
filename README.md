# AGDIFF_chi

AGDIFF_chi is an all-atom diffusion workflow for molecular 3D structure generation with stereochemistry-aware conformer filtering. This repository contains the AGDIFF codebase plus a cyclic-peptide/small-molecule generation entry point that was used for the stereogenic implementation described in the JCIM paper below.

**Paper:** [Accurate 3D Structure Prediction of Small Cyclic Peptides Containing Non-Canonical Amino Acid Residues Using an All-Atom Diffusion Model with Stereogenic Implementation](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c03236), *Journal of Chemical Information and Modeling*, 2026.

<p align="center">
  <img src="assets/diffusion.gif" alt="AGDIFF_chi molecule generation animation" width="80%">
</p>

## What is included

- `scripts/smiles_generation.py` — SMILES-to-SDF generation script.
- `logs/cremp_default_batch64_2024_12_12__14_42_15/best_model/best_model.pt` — bundled pretrained checkpoint used for the AGDIFF_chi code test.
- `logs/cremp_default_batch64_2024_12_12__14_42_15/cremp_default_batch64.yml` — paired configuration file loaded with the checkpoint.
- `assets/diffusion.gif` — molecule-generation animation.
- `agdiff.yml` — Conda environment file for a public AGDIFF_chi installation.
- `setup.py` / `pyproject.toml` — editable-install metadata inherited from the original AGDIFF package layout.

## Robust stereochemistry filtering

### Released checkpoint compatibility

The bundled CREMP checkpoint was trained with **`edge_encoder_global` feeding both
the global and local branches**. Using `edge_encoder_local` in the local branch
loads without a state-dict error, but is not the computation used by this checkpoint.
This has been corrected: absent `model.local_edge_encoder`, the model now uses the
shared global edge encoder. The local module remains registered so existing
checkpoint keys still load strictly. Only checkpoints explicitly trained with
independent edge embeddings should set `model.local_edge_encoder: local` in their
embedded model configuration.

A controlled 5,000-step test with the released weights and identical seeds gave
**0/32 bond-QC passes for each of cyclo(L-Ala)5 and cyclo(L-Ala)6** with the previous
routing, versus **32/32 for each** with the corrected routing. Bond QC here means
finite coordinates and every covalent bond strictly between 0.8 and 2.0 Å; it is a
gross geometry screen, not an energy or conformational-accuracy benchmark.

The `scripts/smiles_generation.py` workflow includes robust post-generation filtering for molecules with specified stereocenters:

1. Compare only the chiral centers that are explicitly specified in the input SMILES.
   Assign these centers freshly from 3D coordinates after removing template
   stereochemistry. Missing/undefined assignments or failed perception are rejected.
2. Ignore newly assigned R/S labels at originally unspecified centers.
3. If every originally specified chiral center is reversed, treat the conformer as a global mirror image, reflect its coordinates, re-detect 3D chirality, and keep it only if the reflected conformer matches the target stereochemistry.
4. Apply a simple covalent bond-length QC before accepting a conformer.
5. Stop early once the requested number of filtered conformers has been obtained.
6. Write the final filtered output as `molecules.sdf` in the requested output directory.

Accepted conformers are annotated in the SDF with properties such as:

- `agdiff_source_final_index`
- `agdiff_filter_reason`
- `agdiff_min_bond_A`
- `agdiff_max_bond_A`

Common filter reasons include:

- `specified_centers_match`
- `all_specified_centers_reversed_flipped_to_match`

Undefined stereochemistry is **not** treated as flippable. Only a fully assigned
global mirror image can be reflected, and it must pass a fresh post-reflection check.

## Environment setup

The recommended setup follows the original AGDIFF installation pattern: create a Conda environment, install PyTorch/PyG packages matched to your CUDA build, then install this repository in editable mode.

### 1. Create the Conda environment

```bash
conda env create -f agdiff.yml
conda activate agdiff
```

The provided `agdiff.yml` targets Python 3.10 with PyTorch 2.4.x and CUDA 12.1. It includes the common scientific dependencies used by AGDIFF/AGDIFF_chi, including RDKit, NumPy, SciPy, pandas, scikit-learn, PyYAML, EasyDict, tqdm, TensorBoard, NetworkX, and Joblib.

If your workstation uses a different CUDA version, adjust the PyTorch/PyG commands below according to the official PyTorch and PyTorch Geometric installation pages.

### 2. Install PyTorch Geometric compiled extensions

For the CUDA 12.1 / PyTorch 2.4 environment above, install the PyG packages with the matching wheel index:

```bash
pip install torch_geometric==2.6.1
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

### 3. Install AGDIFF_chi in editable mode

From the repository root:

```bash
pip install -e .
```

### 4. Verify the installation

```bash
python - <<'PY'
import torch
import torch_geometric
import torch_scatter
import torch_sparse
import torch_cluster
import rdkit

print('torch', torch.__version__)
print('torch_geometric', torch_geometric.__version__)
print('torch_scatter', torch_scatter.__version__)
print('torch_sparse', torch_sparse.__version__)
print('torch_cluster', torch_cluster.__version__)
print('rdkit', rdkit.__version__)
print('cuda_available', torch.cuda.is_available())
PY
```

For GPU generation, `cuda_available` should be `True`.

## Generate conformers from a SMILES string

### Target-driven filtered generation (recommended)

`generate_filtered` samples batches until the **required number of accepted
conformers** is reached or the automatic raw-candidate cap is exhausted.
One process uses one device. Run it
from the repository root in the installed environment:

```bash
CKPT=logs/cremp_default_batch64_2024_12_12__14_42_15/best_model/best_model.pt
ALA5='C[C@@H]1NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](C)NC1=O'
ALA6='C[C@@H]1NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](C)NC1=O'

python -m scripts.generate_filtered "$CKPT" --smiles "$ALA5" \
  --out outputs/ala5 --target 100 --batch-size 128 \
  --seed 202609270 --n-steps 5000

python -m scripts.generate_filtered "$CKPT" --smiles "$ALA6" \
  --out outputs/ala6 --target 100 --batch-size 128 \
  --seed 202609271 --n-steps 5000
```

- `--target` must be positive. Generation stops automatically after the first
  completed batch that meets the target; `--candidates` and `--early-stop` have
  been removed. The last raw batch is capped to the remaining raw allowance.
- The raw upper limit is `int(target * 1.2 * 2**n)`, where `n` is the number of
  RDKit chiral centers in the input graph, **including unassigned centers**:
  `Chem.FindMolChiralCenters(base, includeUnassigned=True, useLegacyImplementation=False)`.
  This counts atom chiral centers, not E/Z double bonds. It changes only the
  oversampling allowance; QC still compares only explicitly specified centers.
  An achiral molecule uses `n=0`. The examples above have caps of **3,840** and
  **7,680** raw candidates, respectively, not guaranteed accepted counts.
- A shortfall writes its summary and exits with status **2**, rather than claiming
  success. Generate a new independently seeded run if needed; do not relax QC to
  fill the target. A successful run exits with status **0**.
- `molecules.sdf` contains at most the requested target. `summary.json` distinguishes
  generated, accepted, rejected, and written counts and records sampler settings,
  seed, checkpoint hash, batch sizes, and CUDA peak allocated/reserved memory.
  Accepted counts include all QC passes in completed batches; written counts are
  capped at the target. Summary and progress metadata include `chiral_center_count`,
  `candidate_budget` (the automatic cap), and `stop_reason` (`target_met` or
  `candidate_budget_exhausted`; intermediate progress uses `running`). The summary
  also records the explicit `chiral_center_policy`.
- For parallel sampling, assign a **positive accepted target per shard**, one
  process per scheduler-assigned GPU, distinct seeds, and separate output
  directories. The old `--target 0` collect-all shard mode is no longer supported.
  Reserve non-overlapping `--candidate-offset` ranges using each shard's full
  automatic cap, not its eventual generated count. For equal shard targets and
  the same molecule, shard `i` can use offset `i * int(shard_target * 1.2 * 2**n)`.
  Each shard stops independently and may exit 2 with a shortfall. Merge retained
  accepted records and independently validate before selecting the final target;
  a shard shortfall must not be reported as global success without that count gate.
- `--batch-size` is a candidate count, not a reference multiplier. Start conservatively
  and measure peak VRAM over the full trajectory. An OOM splits only the failed
  batch and records the retry; it does not silently reduce the raw cap. Future
  batches are scheduled lazily, so the exponential cap does not allocate a huge
  queue. Pending retries are abandoned when the accepted target is reached.
- Full coordinate histories are disabled in this entry point. This changes storage,
  not the sampled coordinates. No force-field optimization is performed.
- Existing nonempty output directories are not overwritten. Use a new directory for
  every run. Only load trusted checkpoints: the bundled historical checkpoint
  includes Python objects and is loaded with `weights_only=False`.

### Legacy entry point

The original command below remains available. Its target is
`num_confs * num_refs`; its candidate budget also includes stereocenter oversampling
and a 1.2 safety factor. **`--gpus` only divides that budget; it does not launch
multiple GPU workers.** Prefer the target-driven entry point above for sharding
and a checked acceptance target.

Example: generate five filtered conformers for cyclo(Ala-Ala-Ala-Ala-Ala).

```bash
CKPT=logs/cremp_default_batch64_2024_12_12__14_42_15/best_model/best_model.pt
SMILES='C[C@@H]1NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](C)NC1=O'

python scripts/smiles_generation.py "$CKPT" \
  --smiles "$SMILES" \
  --out_sdf outputs/cyclo_AAAAA \
  --num_confs 1 \
  --num_refs 5 \
  --max_num_refs 50 \
  --gpus 1 \
  --n_steps 5000 \
  --tag cyclo_AAAAA_demo
```

The filtered result is written to:

```text
outputs/cyclo_AAAAA/molecules.sdf
```

## Output interpretation

During generation, the script may oversample because stereochemical filtering can reject many raw candidates. For example, in a five-conformer cyclo(Ala)5 test, the script generated 98 raw conformers in two chunks of 49 raw conformers each, then early-stopped after five conformers passed stereochemistry and bond-length QC.

Use RDKit to count accepted records:

```bash
python - <<'PY'
from rdkit import Chem
path = 'outputs/cyclo_AAAAA/molecules.sdf'
print(sum(1 for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None))
PY
```

## Regression tests

Run in the AGDIFF environment from the repository root:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Tests cover checkpoint/encoder routing, trajectory-storage parity, strict 3D
stereochemistry (including planar, mirrored and mixed-chirality controls), nonfinite
geometry, target-driven stopping, assigned/unassigned center caps, lazy scheduling,
tail batches, target shortfalls, OOM recovery, large metadata IDs and overwrite
protection. Target-loop control tests use an explicitly synthetic sampler with real
RDKit I/O/QC; metadata and OOM integration tests use a tiny real CPU model. These
tests are not a large-GPU generation or conformational-quality benchmark.

## Original AGDIFF usage

Training and benchmark evaluation scripts from the base AGDIFF implementation are kept in this repository. Example training commands:

```bash
python scripts/train.py ./configs/qm9_default.yml
python scripts/train.py ./configs/drugs_default.yml
```

Example benchmark generation command:

```bash
python scripts/test.py ./logs/path/to/checkpoints/${iter}.pt ./configs/qm9_default.yml \
  --start_idx 0 --end_idx 200
```

## Citation

If you use AGDIFF_chi, please cite:

```bibtex
@article{wu2026agdiffchi,
  title   = {Accurate 3D Structure Prediction of Small Cyclic Peptides Containing Non-Canonical Amino Acid Residues Using an All-Atom Diffusion Model with Stereogenic Implementation},
  author  = {Wu, Dizhou and Zou, Yike},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2026},
  doi     = {10.1021/acs.jcim.5c03236}
}
```

This repository builds on the original AGDIFF implementation:

```bibtex
@misc{wyzykowskiAGDIFFAttentionEnhancedDiffusion2024,
  title         = {{{AGDIFF}}: {{Attention-Enhanced Diffusion}} for {{Molecular Geometry Prediction}}},
  author        = {Wyzykowski, Andr{\'e} Brasil Vieira and Fathi Niazi, Fatemeh and Dickson, Alex},
  year          = {2024},
  month         = oct,
  publisher     = {ChemRxiv},
  doi           = {10.26434/chemrxiv-2024-wrvr4},
  archiveprefix = {ChemRxiv}
}
```

## Acknowledgement

AGDIFF_chi is based on AGDIFF, GEODIFF, PyTorch, PyTorch Geometric, and SchNet. We thank the developers and contributors of these projects for making their work available.
