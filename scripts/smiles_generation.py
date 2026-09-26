import os
import argparse
import pickle
import yaml
import torch
import numpy as np
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict

from src.agdiff.models.epsnet import get_model
from src.agdiff.utils.datasets import PackedConformationDataset
from src.agdiff.utils.transforms import Compose, CountNodesPerGraph, AddHigherOrderEdges
from src.agdiff.utils.misc import get_new_log_dir, seed_all, repeat_data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, remove_self_loops
from src.agdiff.models.common import _extend_graph_order

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType
import copy
import random
import math
from torch_scatter import scatter
from torch_geometric.data import Data
from src.agdiff.utils.chem import BOND_TYPES
from torch_sparse import coalesce

def num_confs(num: str):
    print(f"Parsing num_confs argument: {num}")
    if num.endswith('x'):
        multiplier = int(num[:-1])
        print(f"num_confs ends with 'x', multiplier set to: {multiplier}")
        return lambda x: x * multiplier
    elif num.isdigit() and int(num) > 0:
        absolute = int(num)
        print(f"num_confs is a positive integer, absolute count set to: {absolute}")
        return lambda x: absolute
    else:
        raise ValueError(f"Invalid num_confs value: {num}")

def rdmol_to_data(mol: Chem.Mol, smiles=None):
    assert mol.GetNumConformers() == 1
    print("Converting RDKit molecule to PyTorch Geometric Data object...")
    N = mol.GetNumAtoms()
    print(f"Number of atoms in molecule: {N}")

    # Initialize positions to zeros since we are not using RDKit's conformer positions
    pos = torch.zeros((N, 3), dtype=torch.float32)
    print("Initialized atomic positions to zeros.")

    # Extract atomic properties
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
    print("Extracted atomic properties.")

    z = torch.tensor(atomic_number, dtype=torch.long)
    print("Atomic numbers tensor created.")

    # Extract bond information using BOND_TYPES_MAPPING from utils.chem
    row, col, edge_type = [], [], []
    BOND_TYPES_MAPPING = {bond_type: idx for idx, bond_type in enumerate(BOND_TYPES)}
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bond_type = bond.GetBondType()
        bond_type_idx = BOND_TYPES_MAPPING.get(bond_type, 0)
        edge_type += 2 * [bond_type_idx]
    print("Extracted bond information.")

    # Convert bond information to tensors
    try:
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        print("Edge index and bond types tensors created.")
    except Exception as e:
        print(f"Error converting bond information to tensors: {e}")
        raise

    # Sort edge indices and bond types
    try:
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
        print("Sorted edge indices and bond types.")
    except Exception as e:
        print(f"Error sorting edge indices and bond types: {e}")
        raise

    # Coalesce to ensure unique edges
    print("Coalescing edge_index and edge_type to ensure unique edges...")
    try:
        edge_index, edge_type = coalesce(edge_index, edge_type, N, N)
        print(f"After coalesce: edge_index shape: {edge_index.shape}, edge_type shape: {edge_type.shape}")
    except Exception as e:
        print(f"Error during coalesce of edge_index and edge_type: {e}")
        raise

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)
        print(f"Generated SMILES from molecule: {smiles}")

    # Create PyTorch Geometric Data object without including 'rdmol' to avoid segmentation faults
    try:
        data = Data(
            atom_type=z,
            pos=pos,
            edge_index=edge_index,
            edge_type=edge_type,
            smiles=smiles
        )
        print("Data object created successfully.")
    except Exception as e:
        print(f"Error creating Data object: {e}")
        raise

    return data

def _agdiff_bond_stats(mol):
    """Return finite bond statistics, or None for missing/invalid geometry."""
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer(0)
    # Check every atom, including disconnected atoms not visited by the bonds.
    if not np.isfinite(conf.GetPositions()).all():
        return None
    distances = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pi, pj = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
        distance = math.dist((pi.x, pi.y, pi.z), (pj.x, pj.y, pj.z))
        if not math.isfinite(distance):
            return None
        distances.append(distance)
    if not distances:
        return None
    return min(distances), max(distances), sum(distances) / len(distances)


def _agdiff_3d_chiral_centers(mol):
    """Infer centers only from finite 3D coordinates; return None on failure.

    An empty dictionary denotes successful perception of an achiral molecule,
    not a perception failure. Never fall back to the tagged template.
    """
    try:
        if mol.GetNumConformers() == 0:
            return None
        probe = Chem.Mol(mol)
        conf = probe.GetConformer(0)
        if not np.isfinite(conf.GetPositions()).all():
            return None
        Chem.RemoveStereochemistry(probe)
        conf.Set3D(True)
        Chem.AssignAtomChiralTagsFromStructure(probe, conf.GetId(), True)
        Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
        return dict(Chem.FindMolChiralCenters(probe, includeUnassigned=True))
    except Exception as exc:
        print(f"WARNING: 3D chiral-center perception failed; rejecting molecule: {exc}")
        return None


def _agdiff_reflect_mol_coordinates(mol):
    """Return a coordinate-mirrored copy, which flips all stereocenters.

    Reflection is done around the conformer's x centroid to preserve scale and
    position while changing handedness.
    """
    flipped = Chem.Mol(mol)
    if flipped.GetNumConformers() == 0:
        return flipped
    conf = flipped.GetConformer(0)
    xs = [conf.GetAtomPosition(i).x for i in range(flipped.GetNumAtoms())]
    cx = sum(xs) / len(xs)
    for i in range(flipped.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(2.0 * cx - p.x, p.y, p.z))
    return flipped


def _agdiff_target_chiral_centers(smiles):
    """Return target centers and the subset explicitly specified in SMILES."""
    target = Chem.MolFromSmiles(smiles)
    if target is None:
        raise ValueError(f"Invalid SMILES string for chiral filtering: {smiles}")
    target_h = Chem.AddHs(target)
    centers = dict(Chem.FindMolChiralCenters(target_h, includeUnassigned=True))
    specified = {idx: val for idx, val in centers.items() if val != '?'}
    return centers, specified


def _agdiff_chirality_matches(mol, specified_centers):
    """Require fresh R/S assignments at every specified target center."""
    centers = _agdiff_3d_chiral_centers(mol)
    if centers is None:
        return False, {}, 'chiral_perception_failed'

    opposite = {'R': 'S', 'S': 'R'}
    for idx, expected in specified_centers.items():
        if expected not in opposite or centers.get(idx) not in opposite:
            return False, centers, f'specified_center_{idx}_unassigned'
    if specified_centers and all(
        expected in opposite and centers.get(idx) == opposite[expected]
        for idx, expected in specified_centers.items()
    ):
        return False, centers, 'all_specified_centers_reversed'

    for idx, expected in specified_centers.items():
        if centers.get(idx) != expected:
            return False, centers, f'specified_center_{idx}_mismatch'
    return True, centers, 'specified_centers_match'


def filter_sdf_by_chirality(sdf_file, output_sdf, smiles, num_needed,
                            min_bond=0.8, max_bond=2.0):
    """Write at most num_needed conformers passing strict chirality/bond QC.

    Always creates output_sdf. Every specified center must have a matching R/S
    assignment from finite coordinates; missing or undefined assignments fail.
    Fully reversed conformers may be reflected only with a successful recheck.
    """
    target_centers, specified_centers = _agdiff_target_chiral_centers(smiles)
    print(f"Target chiral centers: {target_centers}")
    print(f"Specified centers used for filtering: {specified_centers}")
    os.makedirs(os.path.dirname(output_sdf) or '.', exist_ok=True)
    writer = Chem.SDWriter(output_sdf)
    saved_mols = 0
    saved_mol_indices = []
    total_mols = 0
    rejected_chi = 0
    rejected_bond = 0
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    for i, cand in enumerate(suppl):
        if saved_mols >= num_needed:
            break
        if cand is None:
            continue
        total_mols += 1
        chi_ok, centers, reason = _agdiff_chirality_matches(cand, specified_centers)
        if not chi_ok and reason == 'all_specified_centers_reversed':
            flipped = _agdiff_reflect_mol_coordinates(cand)
            flip_ok, flip_centers, flip_reason = _agdiff_chirality_matches(flipped, specified_centers)
            if flip_ok:
                cand = flipped
                centers = flip_centers
                reason = 'all_specified_centers_reversed_flipped_to_match'
                chi_ok = True
                print(f"Flipped all-reversed molecule {i}; post-flip reason={flip_reason}")
            else:
                print(f"WARNING: molecule {i} looked all-reversed but did not pass after reflection: {flip_reason}, centers={flip_centers}")
        stats = _agdiff_bond_stats(cand)
        bond_ok = stats is not None and stats[0] > min_bond and stats[1] < max_bond
        if not chi_ok:
            rejected_chi += 1
            continue
        if not bond_ok:
            rejected_bond += 1
            continue
        cand.SetProp('agdiff_filter_reason', reason)
        cand.SetProp('agdiff_chiral_centers', str(centers))
        cand.SetDoubleProp('agdiff_min_bond_A', float(stats[0]))
        cand.SetDoubleProp('agdiff_max_bond_A', float(stats[1]))
        cand.SetDoubleProp('agdiff_mean_bond_A', float(stats[2]))
        cand.SetIntProp('agdiff_source_final_index', int(i))
        writer.write(cand)
        saved_mols += 1
        saved_mol_indices.append(i)
        print(f"Accepted molecule {i}: reason={reason}, bond_min={stats[0]:.3f}, bond_max={stats[1]:.3f}")
        if saved_mols >= num_needed:
            break
    writer.close()
    print(f"Filter summary: total_seen={total_mols}, saved={saved_mols}, rejected_chirality={rejected_chi}, rejected_bond={rejected_bond}, output={output_sdf}")
    print(f"Indices of saved molecules: {saved_mol_indices}")
    if saved_mols == 0:
        print("WARNING: Filter.sdf was created but no conformers passed the strict chirality + bond QC filter.")
    return saved_mols, saved_mol_indices


def _agdiff_write_positions_to_sdf(template_mol, pos_gen, num_nodes, output_sdf):
    """Write a tensor shaped [num_confs, num_nodes, 3] to a multi-record SDF."""
    out_mol = Chem.Mol(template_mol)
    out_mol.RemoveAllConformers()
    for i in range(pos_gen.shape[0]):
        conf = Chem.Conformer(num_nodes)
        positions = pos_gen[i]
        for atom_idx in range(num_nodes):
            x, y, z = positions[atom_idx].tolist()
            conf.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(x, y, z))
        conf.SetId(int(i))
        out_mol.AddConformer(conf, assignId=True)
    writer = Chem.SDWriter(output_sdf)
    for conf in out_mol.GetConformers():
        writer.write(out_mol, confId=conf.GetId())
    writer.close()
    print(f"Wrote {out_mol.GetNumConformers()} conformers to {output_sdf}")
    return output_sdf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate conformers from SMILES and save as SDF.")
    parser.add_argument('ckpt', type=str, help='Path for loading the checkpoint')
    parser.add_argument('--smiles', type=str, required=True, help='Input SMILES string')
    parser.add_argument('--out_sdf', type=str, required=True, help='Output SDF file path')
    parser.add_argument('--save_traj', action='store_true', default=False,
                        help='Whether to store the whole trajectory for sampling')
    parser.add_argument('--num_confs', type=int, default=2)
    parser.add_argument('--num_refs', type=int, default=100)  
    parser.add_argument('--max_num_refs', type=int, default=200)  
    parser.add_argument('--gpus', type=int, default=1)  
    parser.add_argument('--tag', type=str, default='', help='Tag for the output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: "cuda" or "cpu"')
    parser.add_argument('--clip', type=float, default=1000.0, help='Clipping value for gradients')
    parser.add_argument('--n_steps', type=int, default=5000,
                        help='Number of sampling steps')
    parser.add_argument('--global_start_sigma', type=float, default=0.5,
                        help='Enable global gradients only when noise is low')
    parser.add_argument('--w_global', type=float, default=1.0, help='Weight for global gradients')
    parser.add_argument('--sampling_type', type=str, default='ld',
                        help='Sampling method: generalized, ddpm_noisy, ld')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='Weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    args = parser.parse_args()

    print("Starting generate_conformer.py script...")
    print(f"Parsed arguments: {args}")

    # Load checkpoint
    print("Loading checkpoint...")
    try:
        ckpt = torch.load(args.ckpt, map_location=args.device)
        print(f"Checkpoint loaded successfully from {args.ckpt}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)

    # Load configuration
    config_path = glob(os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), '*.yml'))
    if len(config_path) == 0:
        print("Configuration YAML file not found.")
        exit(1)
    config_path = config_path[0]
    print(f"Loading configuration from {config_path}...")
    try:
        with open(config_path, 'r') as f:
            config = EasyDict(yaml.safe_load(f))
        print("Configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit(1)

    # Set random seed for reproducibility
    print("Setting random seed...")
    try:
        seed_random = random.randint(0, 1000000)
        print("Running with SEED: ", seed_random)
        seed_all(seed_random)
    except Exception as e:
        print(f"Error setting random seed: {e}")
        exit(1)

    # Determine log/output directory
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))
    output_dir = get_new_log_dir(log_dir, 'sample', tag=args.tag)
    print(f"Output directory is set to: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created or already exists: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        exit(1)

    # Model loading
    print('Loading model...')
    try:
        model = get_model(ckpt['config'].model).to(args.device)
        print("Model instantiated successfully.")
        model.load_state_dict(ckpt['model'])
        print("Model state loaded successfully.")
        model.eval()
        print("Model set to evaluation mode.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Process the SMILES string
    smiles = args.smiles
    print(f"Processing SMILES string: {smiles}")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")
        print("SMILES string parsed into RDKit molecule successfully.")
    except Exception as e:
        print(f"Failed to parse SMILES string '{smiles}': {e}")
        exit(1)

    # Add hydrogens
    print("Adding hydrogens to the molecule...")
    try:
        mol = Chem.AddHs(mol)
        print("Hydrogens added to the molecule successfully.")
        num_chi = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        print('chi:', num_chi)
    except Exception as e:
        print(f"Error adding hydrogens: {e}")
        exit(1)




        
    # Generate an initial 3D conformer
    print('Generating initial 3D conformer...')
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed_random   # Use the same random seed for reproducibility
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            raise ValueError("Conformer generation failed.")
        print("Initial 3D conformer generated successfully.")
    except Exception as e:
        print(f"Failed to generate initial conformer: {e}")
        exit(1)

    if mol.GetNumConformers() == 0:
        print("No conformers found in the molecule after embedding.")
        exit(1)
    else:
        print(f"Number of conformers in the molecule: {mol.GetNumConformers()}")

    # Convert RDKit molecule to data object
    print("Converting molecule to data object...")
    try:
        data_input = rdmol_to_data(mol)
        print("Molecule converted to data object successfully.")
    except Exception as e:
        print(f"Error converting molecule to data object: {e}")
        exit(1)

    # Apply transforms
    print("Applying transforms to data object...")
    transforms = Compose([
        CountNodesPerGraph(),
        AddHigherOrderEdges(order=config.model.edge_order)  # Offline edge augmentation
    ])

    try:
        data_input = transforms(data_input)
        print("Transforms applied successfully.")
    except Exception as e:
        print(f"Error applying transforms: {e}")
        exit(1)

    num_nodes = data_input.num_nodes
    print(f"Number of atoms in the molecule: {num_nodes}")

    # Determine number of conformers to generate
    print("Determining number of conformers to generate...")
    num_samples_saved = args.num_confs * args.num_refs
    num_refs = int(args.num_refs * 2**num_chi * 1.2 / args.gpus)
    num_samples = args.num_confs * num_refs
    print(f"Number of references (num_refs): {num_refs}")
    print(f"Number of conformers to generate (num_samples): {num_samples}")

    
    # If num_refs is within limit, process normally
    max_num_refs = min(args.max_num_refs, 26000 // data_input.num_nodes) if args.max_num_refs and args.max_num_refs > 0 else (26000 // data_input.num_nodes)
    print("Max number of references:", max_num_refs)
    if num_refs < max_num_refs+1:

        # Prepare batch
        print("Preparing batch data...")
        data_input.pos_ref = None
        try:
            batch = repeat_data(data_input, num_samples).to(args.device)
            print("Data repeated and moved to the specified device successfully.")
        except Exception as e:
            print(f"Error preparing batch data: {e}")
            exit(1)
     
        # Sampling
        print("Starting conformer sampling...")
        clip_local = None
        success = False
        results = []
        done_smiles = set()
        for attempt in range(2):  # Maximum number of retries
            print(f"Sampling conformations (Attempt {attempt + 1})...")
            try:
                pos_init = torch.randn(batch.num_nodes, 3).to(args.device)
                print("Initialized random positions for sampling.")
                with torch.no_grad():
                    pos_gen, pos_gen_traj = model.langevin_dynamics_sample_diffusion(
                        atom_type=batch.atom_type,
                        pos_init=pos_init,
                        bond_index=batch.edge_index,
                        bond_type=batch.edge_type,
                        batch=batch.batch,
                        num_graphs=batch.num_graphs,
                        extend_order=False,  # Done in transforms
                        n_steps=args.n_steps,
                        step_lr=1e-6,
                        w_global=args.w_global,
                        global_start_sigma=args.global_start_sigma,
                        clip=args.clip,
                        clip_local=clip_local,
                        sampling_type=args.sampling_type,
                        eta=args.eta
                    )
                pos_gen = pos_gen.cpu()
                print("Conformations sampled successfully.")
     
                if args.save_traj:
                    data_input.pos_gen = torch.stack(pos_gen_traj)
                else:
                    data_input.pos_gen = pos_gen
                results.append(data_input)
                done_smiles.add(data_input.smiles)
     
                save_path = os.path.join(output_dir, 'samples_all.pkl')
                print(f'Saving samples to: {save_path}')
                with open(save_path, 'wb') as f:
                    pickle.dump(results, f)
     
                success = True
                break  # Break the retry loop if successful
            except FloatingPointError:
                clip_local = 20
                print('FloatingPointError encountered. Retrying with local clipping.')
            except Exception as e:
                print(f"Error during sampling: {e}")
                exit(1)
     
        if not success:
            print("Sampling failed after retries.")
            exit(1)


        # Reshape generated positions
        print("Reshaping generated positions...")
        print(pos_gen.shape)
     
        try:
            pos_gen = pos_gen.view(num_samples, num_nodes, 3)
            print(f"Generated positions reshaped to ({num_samples}, {num_nodes}, 3).")
        except Exception as e:
            print(f"Error reshaping generated positions: {e}")
            exit(1)
     
        # Update RDKit molecule with generated conformers
        print("Updating molecule with generated conformers...")
        try:
            mol.RemoveAllConformers()            
            print("Removed all existing conformers from the molecule.")
            stride=1
            for i in range(0,pos_gen.shape[0],stride):
                conf = Chem.Conformer(num_nodes)
                positions = pos_gen[i]  # Shape: (num_nodes, 3)
                for atom_idx in range(num_nodes):
                    x, y, z = positions[atom_idx].tolist()
                    conf.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(x, y, z))
                    
                conf.SetId(i)
                mol.AddConformer(conf, assignId=True)
                print('Chiral:', Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                print(f"Added conformer {i} to the molecule.")
            print(f"Total of {num_samples} conformers added to the molecule.")
        except Exception as e:
            print(f"Error updating molecule with conformers: {e}")
            exit(1)


            
        # Write the molecule with conformers to an SDF file
        print("Writing conformers to SDF file...")
        try:
            # Ensure the directory exists
            os.makedirs(args.out_sdf, exist_ok=True)
     
            # Construct the full output file path
            output1 = os.path.join(args.out_sdf, ".agdiff_candidates_tmp.sdf")
            
            # Write conformers to the specified SDF file
            writer = Chem.SDWriter(output1)
            for conf in mol.GetConformers():
                writer.write(mol, confId=conf.GetId())
                print(f"Conformer {conf.GetId()} written to SDF.")
            writer.close()
            print(f"All generated conformers saved to {args.out_sdf} successfully.")
        except Exception as e:
            print(f"Error writing to SDF file: {e}")
            exit(1)


        # Robust relaxed chirality + bond-QC filter. Always creates *_Filter.sdf.
        output2 = os.path.join(args.out_sdf, "molecules.sdf")
        saved_mols, saved_mol_indices = filter_sdf_by_chirality(
            output1, output2, smiles, num_samples_saved)
        try:
            os.remove(output1)
        except OSError:
            pass



        


            

            
        if args.save_traj:
            try:
                pos_gen_traj0 = torch.stack(pos_gen_traj)
                num_frames = pos_gen_traj0.shape[0]
                pos_gen_traj0 = pos_gen_traj0.view(int(num_frames), int(num_samples), num_nodes, 3)
                print('zwm')
                print(pos_gen_traj0.shape)
                print(pos_gen_traj0.shape[0])
                print(f"Generated positions reshaped to ({args.n_steps}, {num_samples}, {num_nodes}, 3).")
            except Exception as e:
                print(f"Error reshaping generated positions: {e}")
                exit(1)
        else:
     
            try:
                num_samples = pos_gen.shape[0] // num_nodes if len(pos_gen.shape) == 2 else pos_gen.shape[0]
                pos_gen = pos_gen.view(num_samples, num_nodes, 3)
                print(f"Generated positions reshaped to ({num_samples}, {num_nodes}, 3).")
            except Exception as e:
                print(f"Error reshaping generated positions: {e}")
                exit(1)
     
        # Update RDKit molecule with generated conformers
        print("Updating molecule with generated conformers...")
        try:
            if args.save_traj:
                stride=10
                for j in range(num_samples):
                    mol.RemoveAllConformers()
                    print("Removed all existing conformers from the molecule.")
                    for i in range(stride-1,num_frames,stride):
                        conf = Chem.Conformer(num_nodes)
                        positions = pos_gen_traj0[i][j]  # Shape: (num_nodes, 3)
                        for atom_idx in range(num_nodes):
                            x, y, z = positions[atom_idx].tolist()
                            conf.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(x, y, z))
                        conf.SetId(i)
                        mol.AddConformer(conf, assignId=True)
                        # Write the molecule with conformers to an SDF file
                        print(f"Writing conformers {i} to SDF file...")
                        try: 
                            # Construct the full output file path
                            output_filename = os.path.join(args.out_sdf, f"{args.out_sdf}{str(j).zfill(10)}.sdf")
                            # Write conformers to the specified SDF file
                            writer = Chem.SDWriter(output_filename)
                            for conf in mol.GetConformers():
                                writer.write(mol, confId=conf.GetId())
                            writer.close()
                        except Exception as e:
                            print(f"Error writing to SDF file: {e}")
                            exit(1)
                    print(f"Conformer {j} saved to {args.out_sdf}{str(j).zfill(10)}.sdf successfully.")    
                print(f"Total of {num_samples} conformers added to the molecule.")
        except Exception as e:
            print(f"Error updating molecule with conformers: {e}")
            exit(1)
     
        # Optionally save the trajectory
        if args.save_traj:
            traj_save_path = os.path.join(output_dir, 'trajectory.pkl')
            print(f"Saving sampling trajectory to {traj_save_path}...")
            try:
                with open(traj_save_path, 'wb') as f:
                    pickle.dump(pos_gen_traj, f)
                print(f"Sampling trajectory saved to {traj_save_path} successfully.")
            except Exception as e:
                print(f"Error saving sampling trajectory: {e}")
                exit(1)
     
        print("Conformer generation process completed successfully.")



















     

     


    else:
        print(f"Number of references exceeds the limit ({max_num_refs}). Processing in chunks...")
        results = []
        done_smiles = set()
        pos_gen_all = []
        pos_gen_traj_all = []
        
        chunks = 1+ (num_refs - 1) // max_num_refs
        chunk_size = 1 + num_refs // chunks
        early_stop = False
        
        for j in range(chunks):
            start = j * chunk_size
            end = min((j + 1) * chunk_size, num_refs)
            num_samples = args.num_confs * (end - start)
            print(f'Processing chunk {j+1}/{chunks}, samples: {start}-{end}, num_samples: {num_samples}')

#            data_input = data.clone()
            data_input["pos_ref"] = None
            batch = repeat_data(data_input, num_samples).to(args.device)
     
            clip_local = None
            for _ in range(2):  # Retry loop
                try:
                    pos_init = torch.randn(batch.num_nodes, 3).to(args.device)
                    pos_gen, pos_gen_traj = model.langevin_dynamics_sample_diffusion(
                        atom_type=batch.atom_type,
                        pos_init=pos_init,
                        bond_index=batch.edge_index,
                        bond_type=batch.edge_type,
                        batch=batch.batch,
                        num_graphs=batch.num_graphs,
                        extend_order=False,
                        n_steps=args.n_steps,
                        step_lr=1e-6,
                        w_global=args.w_global,
                        global_start_sigma=args.global_start_sigma,
                        clip=args.clip,
                        clip_local=clip_local,
                        sampling_type=args.sampling_type,
                        eta=args.eta
                    )
                    # Accumulate in Python lists
                    pos_gen_all.append(pos_gen.cpu())
                    if args.save_traj:
                        pos_gen_traj_all.append(torch.stack(pos_gen_traj).cpu())
                    else:
                        # Cheap boundary early-stop check: write/filter all samples generated so far.
                        os.makedirs(args.out_sdf, exist_ok=True)
                        output1 = os.path.join(args.out_sdf, ".agdiff_candidates_tmp.sdf")
                        output2 = os.path.join(args.out_sdf, "molecules.sdf")
                        current_pos = torch.cat(pos_gen_all, dim=0).view(-1, num_nodes, 3)
                        generated_so_far = int(current_pos.shape[0])
                        print(f"Generated raw conformers so far before filtering: {generated_so_far}")
                        _agdiff_write_positions_to_sdf(mol, current_pos, num_nodes, output1)
                        saved_mols, saved_mol_indices = filter_sdf_by_chirality(
                            output1, output2, smiles, num_samples_saved)
                        try:
                            os.remove(output1)
                        except OSError:
                            pass
                        if saved_mols >= num_samples_saved:
                            print(f"Early stopping after chunk {j+1}/{chunks}: saved {saved_mols}/{num_samples_saved} filtered conformers after generating {generated_so_far} raw conformers.")
                            early_stop = True
                    break
     
                except FloatingPointError:
                    clip_local = 20
                    print('FloatingPointError encountered. Retrying with local clipping.')
            if early_stop:
                break

        # After finishing all chunks, concatenate them
        if len(pos_gen_all) > 0:
            pos_gen_concated = torch.cat(pos_gen_all, dim=0)  # [N, 3]
            if args.save_traj:
                # pos_gen_traj is typically [num_steps, num_nodes, 3] so you might need a different cat strategy.
                # Example if each chunk has same number of steps:
                pos_traj_concated = torch.cat(pos_gen_traj_all, dim=1)  # Concatenate along the node dimension
                data_input.pos_gen = pos_traj_concated
            else:
                data_input.pos_gen = pos_gen_concated
     
        results.append(data_input)  # One entry, with all pos_gen combined
        done_smiles.add(data_input.smiles)










        

        

        
        # Reshape generated positions
        print("Reshaping generated positions...")
        print(pos_gen_concated.shape)
        num_samples = pos_gen_concated.shape[0] // num_nodes
        
        try:
            pos_gen = pos_gen_concated.view(num_samples, num_nodes, 3)
            print(f"Generated positions reshaped to ({num_samples}, {num_nodes}, 3).")
        except Exception as e:
            print(f"Error reshaping generated positions: {e}")
            exit(1)
     
        # Update RDKit molecule with generated conformers
        print("Updating molecule with generated conformers...")
        try:
            mol.RemoveAllConformers()
            print("Removed all existing conformers from the molecule.")
            stride=1
            for i in range(0,pos_gen.shape[0],stride):
                conf = Chem.Conformer(num_nodes)
                positions = pos_gen[i]  # Shape: (num_nodes, 3)
                for atom_idx in range(num_nodes):
                    x, y, z = positions[atom_idx].tolist()
                    conf.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(x, y, z))
                conf.SetId(i)
                mol.AddConformer(conf, assignId=True)
            print(f"Total of {num_samples} conformers added to the molecule.")
        except Exception as e:
            print(f"Error updating molecule with conformers: {e}")
            exit(1)
            
        # Write the molecule with conformers to an SDF file
        print("Writing conformers to SDF file...")
        try:
            # Ensure the directory exists
            os.makedirs(args.out_sdf, exist_ok=True)
     
            # Construct the full output file path
            output1 = os.path.join(args.out_sdf, ".agdiff_candidates_tmp.sdf")
            
            # Write conformers to the specified SDF file
            writer = Chem.SDWriter(output1)
            for conf in mol.GetConformers():
                writer.write(mol, confId=conf.GetId())
                print(f"Conformer {conf.GetId()} written to SDF.")
            writer.close()
            print(f"All generated conformers saved to {args.out_sdf} successfully.")
        except Exception as e:
            print(f"Error writing to SDF file: {e}")
            exit(1)





        # Robust relaxed chirality + bond-QC filter. Always creates *_Filter.sdf.
        output2 = os.path.join(args.out_sdf, "molecules.sdf")
        saved_mols, saved_mol_indices = filter_sdf_by_chirality(
            output1, output2, smiles, num_samples_saved)
        try:
            os.remove(output1)
        except OSError:
            pass















            
        if args.save_traj:
            try:
                pos_gen_traj0 = pos_traj_concated
                num_frames = pos_gen_traj0.shape[0]
                pos_gen_traj0 = pos_gen_traj0.view(int(num_frames), int(num_samples), num_nodes, 3)
                print('zwm')
                print(pos_gen_traj0.shape)
                print(pos_gen_traj0.shape[0])
                print(f"Generated positions reshaped to ({args.n_steps}, {num_samples}, {num_nodes}, 3).")
            except Exception as e:
                print(f"Error reshaping generated positions: {e}")
                exit(1)
        else:
     
            try:
                num_samples = pos_gen.shape[0] // num_nodes if len(pos_gen.shape) == 2 else pos_gen.shape[0]
                pos_gen = pos_gen.view(num_samples, num_nodes, 3)
                print(f"Generated positions reshaped to ({num_samples}, {num_nodes}, 3).")
            except Exception as e:
                print(f"Error reshaping generated positions: {e}")
                exit(1)
     
        # Update RDKit molecule with generated conformers
        print("Updating molecule with generated conformers...")
        try:
            if args.save_traj:
                stride=10
                for j in range(num_samples):
                    mol.RemoveAllConformers()
                    print("Removed all existing conformers from the molecule.")
                    for i in range(stride-1,num_frames,stride):
                        conf = Chem.Conformer(num_nodes)
                        positions = pos_gen_traj0[i][j]  # Shape: (num_nodes, 3)
                        for atom_idx in range(num_nodes):
                            x, y, z = positions[atom_idx].tolist()
                            conf.SetAtomPosition(atom_idx, Chem.rdGeometry.Point3D(x, y, z))
                        conf.SetId(i)
                        mol.AddConformer(conf, assignId=True)
                        # Write the molecule with conformers to an SDF file
                        print(f"Writing conformers {i} to SDF file...")
                        try: 
                            # Construct the full output file path
                            output_filename = os.path.join(args.out_sdf, f"{args.out_sdf}{str(j).zfill(10)}.sdf")
                            # Write conformers to the specified SDF file
                            writer = Chem.SDWriter(output_filename)
                            for conf in mol.GetConformers():
                                writer.write(mol, confId=conf.GetId())
                            writer.close()
                        except Exception as e:
                            print(f"Error writing to SDF file: {e}")
                            exit(1)
                    print(f"Conformer {j} saved to {args.out_sdf}{str(j).zfill(10)}.sdf successfully.")    
                print(f"Total of {num_samples} conformers added to the molecule.")
        except Exception as e:
            print(f"Error updating molecule with conformers: {e}")
            exit(1)
     
        # Optionally save the trajectory
        if args.save_traj:
            traj_save_path = os.path.join(output_dir, 'trajectory.pkl')
            print(f"Saving sampling trajectory to {traj_save_path}...")
            try:
                with open(traj_save_path, 'wb') as f:
                    pickle.dump(pos_gen_traj, f)
                print(f"Sampling trajectory saved to {traj_save_path} successfully.")
            except Exception as e:
                print(f"Error saving sampling trajectory: {e}")
                exit(1)
     
        print("Conformer generation process completed successfully.")



        
