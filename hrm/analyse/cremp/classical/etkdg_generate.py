"""Generate from 2D chemistry only; reference conformers are not an input."""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem


def chiral_codes(mol):
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return {a.GetIdx(): a.GetProp('_CIPCode') for a in mol.GetAtoms() if a.HasProp('_CIPCode')}


def generate(entry, out, candidates, seed, threads):
    start = time.perf_counter()
    identity, smiles = entry['id'], entry['smiles']
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid input SMILES for {identity}')
    expected_stereo = chiral_codes(mol)
    mol = Chem.AddHs(mol)
    assert mol.GetNumConformers() == 0
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = threads
    params.useRandomCoords = True
    params.useMacrocycleTorsions = True
    params.useMacrocycle14config = True
    params.enforceChirality = True
    params.pruneRmsThresh = -1.0  # Keep all proposals; report yield rather than hiding rejection cost.
    params.maxIterations = 1000
    if hasattr(params, 'timeout'):
        params.timeout = 30
    if hasattr(params, 'trackFailures'):
        params.trackFailures = True
    t0 = time.perf_counter()
    returned_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=candidates, params=params))
    # RDKit may return [-1] on timeout although no conformer was created.
    actual_ids = {conf.GetId() for conf in mol.GetConformers()}
    ids = [cid for cid in returned_ids if cid >= 0 and cid in actual_ids]
    if set(ids) != actual_ids:
        raise ValueError(f'Embedding IDs disagree with stored conformers for {identity}')
    embedding_s = time.perf_counter() - t0
    ff_present = bool(AllChem.MMFFHasAllMoleculeParams(mol))
    t0 = time.perf_counter()
    if ids and ff_present:
        opt = list(AllChem.MMFFOptimizeMoleculeConfs(
            mol, numThreads=threads, maxIters=500, mmffVariant='MMFF94'))
    else:
        opt = [(None, None)] * len(ids)
    optimization_s = time.perf_counter() - t0
    conformers = []
    for conf_id, (converged, energy) in zip(ids, opt):
        check = Chem.Mol(mol)
        Chem.AssignStereochemistryFrom3D(check, confId=conf_id, replaceExistingTags=True)
        observed = chiral_codes(check)
        stereo_ok = all(observed.get(i) == code for i, code in expected_stereo.items())
        conformers.append({'id': int(conf_id), 'mmff_status': converged,
                           'mmff_energy_kcal_mol': energy, 'specified_chirality_ok': stereo_ok})
    name = hashlib.sha256(identity.encode()).hexdigest()[:20]
    mol_path = out / f'{name}.pickle'
    mol_path.write_bytes(pickle.dumps(mol, protocol=4))
    elapsed = time.perf_counter() - start
    return {'id': identity, 'input_smiles': smiles, 'input_has_reference_coordinates': False,
            'rdkit_version': rdBase.rdkitVersion, 'seed': seed, 'num_threads': threads,
            'requested_candidates': candidates, 'embedded_candidates': len(ids),
            'embedding_returned_ids': returned_ids, 'embedding_empty': not bool(ids),
            'mmff_available': ff_present, 'conformers': conformers,
            'generation_wall_s': elapsed, 'embedding_wall_s': embedding_s,
            'mmff_wall_s': optimization_s,
            'failure_counts': list(params.GetFailureCounts()) if hasattr(params, 'GetFailureCounts') else None,
            'output_molecule': str(mol_path.resolve()),
            'output_sha256': hashlib.sha256(mol_path.read_bytes()).hexdigest(),
            'selection': 'All embedded candidates retained, including nonconverged optimization; no reference-dependent ranking or filtering.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--candidates', type=int, default=16)
    ap.add_argument('--seed', type=int, default=20261020)
    ap.add_argument('--threads', type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    document = json.loads(args.manifest.read_text())
    entries = document if isinstance(document, list) else document['entries']
    wall, cpu = time.perf_counter(), time.process_time()
    results = []
    for entry in entries:
        # Only these two fields are passed to generation, irrespective of manifest metadata.
        result = generate({'id': entry['id'], 'smiles': entry['smiles']}, args.out,
                          args.candidates, args.seed, args.threads)
        results.append(result)
        (args.out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({'id': result['id'], 'embedded': result['embedded_candidates'],
                          'generation_s': result['generation_wall_s']}), flush=True)
    summary = {'status': 'complete', 'GPU_seconds': 0, 'wall_s': time.perf_counter()-wall,
               'process_CPU_s': time.process_time()-cpu, 'molecules': len(results),
               'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'timing_boundary': 'Imports excluded; generation includes embedding, MMFF, chirality checks, serialization and result writes. Reference evaluation is separate.'}
    (args.out / 'completion.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
