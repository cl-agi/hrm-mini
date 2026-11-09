"""CPU input-isolation gate around unmodified pinned RINGER geometry code.

The fixed angle fixtures are reference-derived plus declared perturbations, not
neural predictions. They are identical across initial-coordinate branches.
This tests information flow, not conformer-generation quality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
ROOT = Path(__file__).resolve().parent
pass
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import scipy
import torch
from ringer.data.macrocycle import MacrocycleAnglesWithSideChainsDataset
from ringer.sidechain_reconstruction import Macrocycle, Reconstructor, set_rdkit_geometries
from ringer.utils import chem, featurization, internal_coords, peptides
FEATURES = MacrocycleAnglesWithSideChainsDataset.feature_names
TOLERANCES = dict(initial_distance_matrix_angstrom=1e-06, reconstructed_distance_matrix_angstrom=0.0001, reconstructed_rtfd=1e-05, full_heavy_distance_matrix_angstrom=0.001)
ORIGINAL_CANONICAL_TRANSFORM = internal_coords.compute_canonical_transform

def proper_canonical_transform(xyz):
    """Small external repair: preserve orientation in the pinned PCA transform."""
    transform = ORIGINAL_CANONICAL_TRANSFORM(xyz)
    if np.linalg.det(transform[:3, :3]) < 0:
        transform[2, :] *= -1
    assert np.linalg.det(transform[:3, :3]) > 0
    return transform

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def single_conformer(mol, conf_id):
    graph = Chem.Mol(mol, quickCopy=True)
    graph.AddConformer(Chem.Conformer(mol.GetConformer(conf_id)), assignId=True)
    return graph

def rebuild_2d(mol):
    """Roundtrip stereo SMILES, recover only the original heavy-atom ordering."""
    graph = Chem.Mol(mol, quickCopy=True)
    for atom in graph.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    text = Chem.MolToSmiles(graph, canonical=True, isomericSmiles=True)
    rebuilt = Chem.MolFromSmiles(text)
    order = sorted(range(rebuilt.GetNumAtoms()), key=lambda i: rebuilt.GetAtomWithIdx(i).GetAtomMapNum())
    rebuilt = Chem.RenumberAtoms(rebuilt, order)
    for atom in rebuilt.GetAtoms():
        atom.SetAtomMapNum(0)
    assert rebuilt.GetNumConformers() == 0
    assert Chem.MolToSmiles(rebuilt) == Chem.MolToSmiles(mol)
    return rebuilt

def embed_2d(graph, seed):
    mol = Chem.AddHs(Chem.Mol(graph, quickCopy=True))
    settings = AllChem.ETKDGv3()
    settings.useRandomCoords = True
    settings.randomSeed = seed
    settings.numThreads = 1
    settings.maxIterations = 1000
    status = AllChem.EmbedMolecule(mol, settings)
    if status != 0:
        raise RuntimeError(f'Independent 2D embedding failed: seed={seed}, status={status}')
    return Chem.RemoveHs(mol)

def graph_field_mask(graph, ring):
    mask = np.zeros((len(ring), len(FEATURES)), dtype=bool)
    mask[:, :2] = True
    torsion_indices = peptides.get_side_chain_torsion_idxs(graph)
    for position, atom_idx in enumerate(ring):
        count = max(0, len(torsion_indices.get(atom_idx, [])) - 3)
        if count > 5:
            raise ValueError('Sidechain exceeds official five-slot representation')
        mask[position, 2:2 + count] = True
        mask[position, 7:7 + count] = True
    return mask

def extract_fixture(mol, ring):
    structures = dict(distance=internal_coords.get_macrocycle_bond_distances(mol, ring), angle=internal_coords.get_macrocycle_bond_angles(mol, ring), dihedral=internal_coords.get_macrocycle_dihedrals(mol, ring), side_chains=internal_coords.get_side_chain_distances_and_angles(mol))
    return MacrocycleAnglesWithSideChainsDataset._flatten_side_chain_features(structures)

def pair_dist(xyz):
    return np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)

def reconstruct_full(graph, ring, result, angles, reconstruction_config):
    sample = {name: pd.DataFrame([angles[:, i]], columns=ring) for i, name in enumerate(FEATURES)}
    for name in ('angle', 'dihedral'):
        sample[name] = pd.DataFrame([result[name].to_numpy()], columns=ring)
    macro = Macrocycle(graph, reconstruction_config, coords=False, copy=True, verify=True)
    assert macro.mol.GetNumConformers() == 0
    model = Reconstructor(macro)
    positions = model.reconstruct(model.parse_internals(sample), model.stacked_tuples)
    mol = set_rdkit_geometries(graph, positions, add_hydrogens=False, copy=True)
    return (mol, positions[0].detach().numpy())

def test_one(path, reconstruction_config, output, seed, normalize_stereo=False):
    started = time.perf_counter()
    with path.open('rb') as handle:
        original = pickle.load(handle)['rd_mol']
    original = Chem.RemoveHs(original)
    reference = single_conformer(original, 0)
    graph = rebuild_2d(reference)
    ring = chem.get_macrocycle_idxs(graph)
    assert ring == chem.get_macrocycle_idxs(reference)
    raw_expected = featurization.featurize_macrocycle_atoms(reference, size=32)
    raw_rebuilt = featurization.featurize_macrocycle_atoms(graph, size=32)
    raw_diffs = [dict(atom=int(raw_expected.index[i]), field=str(raw_expected.columns[j]), reference=int(raw_expected.iloc[i, j]), rebuilt=int(raw_rebuilt.iloc[i, j])) for i, j in zip(*np.where(raw_expected.to_numpy() != raw_rebuilt.to_numpy()))]
    if normalize_stereo:
        Chem.AssignStereochemistry(reference, cleanIt=True, force=True)
        Chem.AssignStereochemistry(graph, cleanIt=True, force=True)
    expected = featurization.featurize_macrocycle_atoms(reference, size=32)
    features_checked = {}
    for name, mol in [('no_conformers', Chem.Mol(reference, quickCopy=True)), ('smiles_rebuilt', graph)]:
        actual = featurization.featurize_macrocycle_atoms(mol, size=32)
        features_checked[name] = bool(expected.equals(actual))
    scrambled = Chem.Mol(reference)
    rng = np.random.default_rng(seed)
    for atom_idx, xyz in enumerate(rng.normal(size=(scrambled.GetNumAtoms(), 3))):
        scrambled.GetConformer().SetAtomPosition(atom_idx, xyz)
    features_checked['random_coordinates'] = bool(expected.equals(featurization.featurize_macrocycle_atoms(scrambled, size=32)))
    fields = extract_fixture(reference, ring)
    fixture = np.stack([fields[name].iloc[0].to_numpy() for name in FEATURES], axis=-1)
    field_mask = graph_field_mask(graph, ring)
    mask_equal = bool(np.array_equal(field_mask, np.isfinite(fixture)))
    fixtures = np.stack([fixture.copy(), fixture.copy()])
    fixtures[1][field_mask] = wrap(fixtures[1][field_mask] + rng.normal(0, 0.03, size=field_mask.sum()))
    inputs = {'reference_first': reference, 'reference_last': single_conformer(original, original.GetNumConformers() - 1)}
    embedding_start = time.perf_counter()
    inputs['dummy_seed1'] = embed_2d(graph, seed + 1)
    inputs['dummy_seed2'] = embed_2d(graph, seed + 2)
    inputs['dummy_seed1_repeat'] = embed_2d(graph, seed + 1)
    embedding_seconds = time.perf_counter() - embedding_start
    for name, mol in inputs.items():
        if normalize_stereo:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        features_checked[name] = bool(expected.equals(featurization.featurize_macrocycle_atoms(mol, size=32)))
    bond_dict = reconstruction_config['macrocycle_ring_distances'][len(ring) // 3]
    distances = pd.Series(np.asarray(bond_dict), index=ring)
    optimizer = internal_coords.RingInternalCoordinates(ring)
    arrays = {'fixed_angles': fixtures, 'graph_field_mask': field_mask, 'ring_indices': np.asarray(ring), 'fixed_bond_distances': distances.to_numpy()}
    results = []
    reference_outputs = {}
    no_coordinates_failure = None
    try:
        optimizer.to_cartesian(graph, distances, pd.Series(fixtures[0, :, 0], index=ring), pd.Series(fixtures[0, :, 1], index=ring), opt_init='average')
    except Exception as exc:
        no_coordinates_failure = f'{type(exc).__name__}: {exc}'
    for sample_idx, values in enumerate(fixtures):
        angle_series = pd.Series(values[:, 0], index=ring)
        torsion_series = pd.Series(values[:, 1], index=ring)
        for name, mol in inputs.items():
            initial = internal_coords.set_macrocycle_geometry_with_average_positions(mol, distances, angle_series, torsion_series, ring)
            initial_xyz = initial.GetConformer().GetPositions()[ring]
            old_transform = ORIGINAL_CANONICAL_TRANSFORM(initial_xyz)
            before_torsion = optimizer.compute_dihedrals(initial_xyz.ravel())
            after_torsion = optimizer.compute_dihedrals(internal_coords.apply_affine_transform(initial_xyz, old_transform).ravel())
            opt_start = time.perf_counter()
            result, status = optimizer.to_cartesian(mol, distances, angle_series, torsion_series, opt_init='average', return_result_obj=True, print_warning=False)
            opt_seconds = time.perf_counter() - opt_start
            side_start = time.perf_counter()
            full_mol, full_xyz = reconstruct_full(graph, ring, result, values, reconstruction_config)
            side_seconds = time.perf_counter() - side_start
            ring_xyz = result[['x', 'y', 'z']].to_numpy()
            torsions = result['dihedral'].to_numpy()
            full_torsions = internal_coords.get_macrocycle_dihedrals(full_mol, ring).iloc[0].to_numpy()
            drift = float(np.abs(wrap(full_torsions - values[:, 1])).mean() / np.pi)
            if name == 'reference_first':
                reference_outputs[sample_idx] = (initial_xyz, ring_xyz, torsions, full_xyz)
            base = reference_outputs[sample_idx]
            row = dict(sample=sample_idx, initialization=name, original_canonical_determinant=float(np.linalg.det(old_transform[:3, :3])), original_canonical_torsion_drift=float(np.abs(wrap(after_torsion - before_torsion)).mean() / np.pi), optimizer_success=bool(status.success), optimizer_message=str(status.message), optimizer_iterations=int(status.nit), optimizer_seconds=opt_seconds, sidechain_seconds=side_seconds, initial_distance_matrix_max_abs=float(np.abs(pair_dist(initial_xyz) - pair_dist(base[0])).max()), reconstructed_distance_matrix_max_abs=float(np.abs(pair_dist(ring_xyz) - pair_dist(base[1])).max()), reconstructed_rtfd_to_reference_initialization=float(np.abs(wrap(torsions - base[2])).mean() / np.pi), full_heavy_distance_matrix_max_abs=float(np.abs(pair_dist(full_xyz) - pair_dist(base[3])).max()), sampled_to_full_rtfd=drift, filter_accept=drift < 0.01, max_ring_bond_constraint_error=float(np.abs(optimizer.compute_distances(ring_xyz.ravel()) - distances).max()))
            results.append(row)
            arrays[f'sample{sample_idx}_{name}_ring_xyz'] = ring_xyz
            arrays[f'sample{sample_idx}_{name}_full_xyz'] = full_xyz
    np.savez(output / f'{path.stem}_outputs.npz', **arrays)
    feature_pass = all(features_checked.values()) and mask_equal
    geometry_pass = all((r['initial_distance_matrix_max_abs'] <= TOLERANCES['initial_distance_matrix_angstrom'] and r['reconstructed_distance_matrix_max_abs'] <= TOLERANCES['reconstructed_distance_matrix_angstrom'] and (r['reconstructed_rtfd_to_reference_initialization'] <= TOLERANCES['reconstructed_rtfd']) and (r['full_heavy_distance_matrix_max_abs'] <= TOLERANCES['full_heavy_distance_matrix_angstrom']) for r in results))
    filter_pass = all((len({r['filter_accept'] for r in results if r['sample'] == i}) == 1 for i in range(2)))
    status_pass = all((len({r['optimizer_success'] for r in results if r['sample'] == i}) == 1 for i in range(2)))
    return dict(molecule=path.stem, source_sha256=sha(path), reference_conformer_count=original.GetNumConformers(), raw_smiles_roundtrip_feature_differences=raw_diffs, candidate_count_fixed=2, ring_atom_count=len(ring), heavy_atom_count=graph.GetNumAtoms(), features_equal=features_checked, graph_mask_equal_to_reference_presence=mask_equal, no_conformer_reconstruction_failure=no_coordinates_failure, embedding_seconds=embedding_seconds, results=results, feature_pass=feature_pass, geometry_pass=geometry_pass, filter_pass=filter_pass, optimizer_status_pass=status_pass, passed=feature_pass and geometry_pass and filter_pass and status_pass, total_seconds=time.perf_counter() - started)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=8)
    parser.add_argument('--normalize-stereo', action='store_true')
    parser.add_argument('--proper-rotation', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    RDLogger.DisableLog('rdApp.warning')
    if args.proper_rotation:
        internal_coords.compute_canonical_transform = proper_canonical_transform
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / 'vendor/ringer/ringer/sidechain_reconstruction/data/reconstruction_data.pickle'
    with config_path.open('rb') as handle:
        reconstruction_config = pickle.load(handle)
    records = []
    for index, path in enumerate(sorted(args.examples.glob('*.pickle'))[:args.limit]):
        record = test_one(path, reconstruction_config, args.output, 20261020 + index * 10, normalize_stereo=args.normalize_stereo)
        records.append(record)
        (args.output / 'per_molecule.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps({'molecule': path.stem, 'passed': record['passed'], 'seconds': record['total_seconds']}), flush=True)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    summary = dict(status='passed' if all((r['passed'] for r in records)) else 'failed', purpose='input-isolation engineering, not generation quality', molecules=len(records), fixed_candidates_per_molecule=2, coordinate_initializations_per_fixture=5, tolerances=TOLERANCES, external_repairs=dict(graph_stereo_normalization=args.normalize_stereo, proper_canonical_rotation=args.proper_rotation), fixtures='first reference conformer plus fixed 0/.03 rad perturbation, same values in all branches', no_model_or_gpu_used=True, gpu_seconds=0, source_modified=False, internal_coords_sha256=sha(internal_coords.__file__), script_sha256=sha(__file__), reconstruction_config_sha256=sha(config_path), reconstruction_constants_provenance='Pinned public RINGER dictionary; full-data geometry statistics, not a 1k-only trained model', runtime=dict(python=sys.version, rdkit=rdkit.__version__, numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__, torch=torch.__version__), wall_seconds=time.perf_counter() - started, cpu_user_seconds=usage.ru_utime, cpu_system_seconds=usage.ru_stime, limits=['Only the selected public L tetrapeptides; no D/NMe/proline or 5/6mer coverage yet', 'Reference-derived angle fixtures test dependence on initial coordinates, not generative quality', 'A no-conformer graph cannot enter unmodified closure; inference must create an independent dummy conformer', 'Passing these finite cases does not prove universal coordinate independence'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
