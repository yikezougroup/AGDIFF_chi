"""Fixed-budget sampling with a separate, strictly filtered acceptance target.

Run from the repository root: python -m scripts.generate_filtered --help
One process uses one scheduler-assigned GPU; use unique seeds, output directories
and candidate offsets for independent shards. No force-field optimization.
"""
import argparse
from collections import Counter, deque
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from rdkit import Chem

from src.agdiff.models.epsnet import get_model
from src.agdiff.utils.misc import repeat_data, seed_all
from src.agdiff.utils.transforms import Compose, CountNodesPerGraph, AddHigherOrderEdges
from scripts.smiles_generation import (
    rdmol_to_data, _agdiff_write_positions_to_sdf, filter_sdf_by_chirality,
)


def batch_sizes(total, size):
    if total <= 0 or size <= 0:
        raise ValueError('Candidate count and batch size must be positive')
    while total:
        take = min(total, size)
        yield take
        total -= take


def target_met(accepted, target):
    return accepted >= target


def prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output directory: {path}')
    path.mkdir(parents=True, exist_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ckpt', type=Path)
    parser.add_argument('--smiles', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--target', type=int, default=100,
                        help='Required accepted count; 0 collects all accepted candidates for a shard')
    parser.add_argument('--candidates', type=int, required=True,
                        help='Raw candidate budget; independent of accepted target and stereocenter count')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--candidate-offset', type=int, default=0)
    parser.add_argument('--n-steps', type=int, default=5000)
    parser.add_argument('--step-lr', type=float, default=1e-6)
    parser.add_argument('--w-global', type=float, default=1.)
    parser.add_argument('--global-start-sigma', type=float, default=.5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--keep-raw', action='store_true')
    parser.add_argument('--early-stop', action='store_true',
                        help='Stop after a completed batch meets target; default samples full budget')
    args = parser.parse_args(argv)
    if args.target < 0 or args.candidate_offset < 0 or args.threads <= 0:
        parser.error('target/offset must be nonnegative and threads positive')
    sizes = deque(batch_sizes(args.candidates, args.batch_size))
    torch.set_num_threads(args.threads)
    prepare_output(args.out)
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    config = checkpoint['config'].model
    if not 1 <= args.n_steps <= config.num_diffusion_timesteps:
        parser.error('n-steps must lie within the checkpoint diffusion schedule')
    model = get_model(config).to(args.device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    seed_all(args.seed)
    base = Chem.MolFromSmiles(args.smiles)
    if base is None or len(Chem.GetMolFrags(base)) != 1:
        parser.error('SMILES must describe one valid connected molecule')
    mol = Chem.AddHs(base)
    # Featurization uses only connectivity/atom types, with zero reference positions.
    # Initial coordinates below are fresh Gaussian noise, not embedded or optimized.
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()))
    data = Compose([CountNodesPerGraph(), AddHigherOrderEdges(order=config.edge_order)])(rdmol_to_data(mol))
    gpu = args.device.startswith('cuda')
    if gpu:
        torch.cuda.reset_peak_memory_stats(args.device)
    counts = Counter(generated=0, accepted=0, written=0, oom_retries=0)
    reasons = Counter()
    batches = []
    start = time.monotonic()
    output_path = args.out / 'molecules.sdf'
    raw_writer = Chem.SDWriter(str(args.out / 'raw.sdf')) if args.keep_raw else None
    writer = Chem.SDWriter(str(output_path))
    try:
        while sizes:
            size = sizes.popleft()
            batch = None
            try:
                batch = repeat_data(data, size)
                batch = batch.to(args.device)
                with torch.no_grad():
                    positions, frames = model.langevin_dynamics_sample_diffusion(
                        atom_type=batch.atom_type,
                        pos_init=torch.randn(batch.num_nodes, 3, device=args.device),
                        bond_index=batch.edge_index, bond_type=batch.edge_type,
                        batch=batch.batch, num_graphs=batch.num_graphs,
                        extend_order=False, n_steps=args.n_steps, step_lr=args.step_lr,
                        w_global=args.w_global, global_start_sigma=args.global_start_sigma,
                        clip=1000., clip_local=None, sampling_type='ld', eta=1., save_traj=False)
                    if not torch.isfinite(positions).all():
                        raise FloatingPointError('Nonfinite sampled coordinates')
                    positions = positions.cpu().reshape(size, mol.GetNumAtoms(), 3)
                assert not frames
            except torch.cuda.OutOfMemoryError:
                del batch
                gc.collect()
                if gpu:
                    torch.cuda.empty_cache()
                if size == 1:
                    raise
                counts['oom_retries'] += 1
                sizes.appendleft(size - size // 2)
                sizes.appendleft(size // 2)
                print(f'OOM: retrying batch {size} as smaller batches', flush=True)
                continue
            del batch
            chunk_raw = args.out / '.chunk_raw.sdf'
            chunk_filtered = args.out / '.chunk_filtered.sdf'
            _agdiff_write_positions_to_sdf(mol, positions, mol.GetNumAtoms(), str(chunk_raw))
            del positions
            accepted, indices = filter_sdf_by_chirality(
                str(chunk_raw), str(chunk_filtered), args.smiles, size)
            if raw_writer:
                for index, candidate in enumerate(Chem.SDMolSupplier(str(chunk_raw), removeHs=False)):
                    if candidate is None:
                        raise ValueError('Generated candidate was not readable after SDF serialization')
                    candidate.SetProp('agdiff_candidate_id', str(args.candidate_offset + counts['generated'] + index))
                    candidate.SetProp('agdiff_seed', str(args.seed))
                    raw_writer.write(candidate)
                raw_writer.flush()
            written_chunk = 0
            if accepted:
                for candidate in Chem.SDMolSupplier(str(chunk_filtered), removeHs=False):
                    if candidate is None:
                        raise ValueError('Filtered candidate was not readable')
                    reasons[candidate.GetProp('agdiff_filter_reason')] += 1
                    if args.target == 0 or counts['written'] < args.target:
                        index = candidate.GetIntProp('agdiff_source_final_index')
                        candidate.SetProp('agdiff_candidate_id', str(args.candidate_offset + counts['generated'] + index))
                        candidate.SetProp('agdiff_seed', str(args.seed))
                        writer.write(candidate)
                        counts['written'] += 1
                        written_chunk += 1
            writer.flush()
            counts['generated'] += size
            counts['accepted'] += accepted
            batches.append({'size': size, 'accepted': accepted, 'written': written_chunk})
            chunk_raw.unlink()
            chunk_filtered.unlink()
            progress = {**counts, 'elapsed_s': time.monotonic() - start, 'batches': batches}
            (args.out / 'progress.json').write_text(json.dumps(progress, indent=2))
            print('PROGRESS', json.dumps(progress), flush=True)
            if args.early_stop and args.target > 0 and target_met(counts['written'], args.target):
                break
    finally:
        writer.close()
        if raw_writer:
            raw_writer.close()
    # Explicit output count gate; no successful completion with a shortfall.
    actual = sum(m is not None for m in Chem.SDMolSupplier(str(output_path), removeHs=False)) if output_path.stat().st_size else 0
    if actual != counts['written']:
        raise ValueError('Output SDF record count does not match generation accounting')
    success = target_met(actual, args.target)
    summary = {
        'success': success, 'target': args.target, 'candidate_budget': args.candidates,
        **counts, 'rejected': counts['generated'] - counts['accepted'],
        'filter_reasons': dict(reasons), 'seed': args.seed, 'candidate_offset': args.candidate_offset,
        'smiles': args.smiles, 'checkpoint_sha256': hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
        'local_edge_encoder': model.local_edge_encoder,
        'n_steps': args.n_steps, 'step_lr': args.step_lr, 'w_global': args.w_global,
        'global_start_sigma': args.global_start_sigma, 'save_traj': False,
        'force_field_optimization': False, 'batch_sizes': batches,
        'elapsed_s': time.monotonic() - start, 'output': str(output_path),
    }
    if gpu:
        torch.cuda.synchronize(args.device)
        summary.update(gpu_name=torch.cuda.get_device_name(args.device),
                       gpu_total_bytes=torch.cuda.get_device_properties(args.device).total_memory,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('RESULT', json.dumps(summary), flush=True)
    return 0 if success else 2


if __name__ == '__main__':
    raise SystemExit(main())
