"""Reference-only metric process using pinned RINGER symmetry-aware metrics."""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from ringer.utils import evaluation


def subset(mol, max_confs):
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    for conf in list(mol.GetConformers())[:max_confs or None]:
        out.AddConformer(Chem.Conformer(conf), assignId=True)
    return out


def scores(matrix, threshold):
    recall_dist, precision_dist = matrix.min(axis=1), matrix.min(axis=0)
    recall = float(np.mean(recall_dist <= threshold))
    precision = float(np.mean(precision_dist <= threshold))
    return {'threshold': threshold, 'cov_recall': recall, 'cov_precision': precision,
            'cov_f1': 2*recall*precision/(recall+precision) if recall+precision else 0.,
            'mat_recall': float(recall_dist.mean()), 'mat_precision': float(precision_dist.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference-manifest', type=Path, required=True)
    ap.add_argument('--generation-results', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-reference-conformers', type=int, default=30)
    ap.add_argument('--cpus', type=int, default=8)
    ap.add_argument('--identity-check', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    refs = json.loads(args.reference_manifest.read_text())
    refs = refs if isinstance(refs, list) else refs['entries']
    refs = {r['id']: r for r in refs}
    generated = json.loads(args.generation_results.read_text())
    t0 = time.perf_counter()
    results = []
    functions = [('all_heavy_rmsd', evaluation.compute_rmsd_matrix, .75),
                 ('ring_rmsd', evaluation.compute_ring_rmsd_matrix, .1),
                 ('ring_tfd', evaluation.compute_ring_tfd_matrix, .05)]
    for gen in generated:
        ref_path = Path(refs[gen['id']]['reference_path'])
        obj = pickle.loads(ref_path.read_bytes())
        reference = obj['rd_mol'] if isinstance(obj, dict) else obj
        reference = subset(reference, args.max_reference_conformers)
        predicted = pickle.loads(Path(gen['output_molecule']).read_bytes())
        entry = {'id': gen['id'], 'num_reference': reference.GetNumConformers(),
                 'num_generated': predicted.GetNumConformers(), 'metrics': {}}
        begin = time.perf_counter()
        if not predicted.GetNumConformers():
            entry['status'] = 'generation_failed'
            for name, _, thresh in functions:
                entry['metrics'][name] = {'threshold': thresh, 'cov_recall': 0.,
                                         'cov_precision': 0., 'cov_f1': 0.,
                                         'mat_recall': None, 'mat_precision': None}
        else:
            entry['status'] = 'ok'
            for name, fn, threshold in functions:
                matrix = fn(Chem.Mol(predicted), Chem.Mol(reference), ncpu=args.cpus)
                entry['metrics'][name] = scores(matrix, threshold)
                matrix_name = hashlib.sha256(gen['id'].encode()).hexdigest()[:20]+'_'+name+'.npz'
                np.savez_compressed(args.out/matrix_name, distances=matrix)
                entry['metrics'][name]['distance_matrix_file'] = matrix_name
                if args.identity_check:
                    one = subset(reference, 1)
                    same = fn(Chem.Mol(one), Chem.Mol(one), ncpu=1)
                    assert float(same.max()) < 1e-6, (name, same)
                    entry.setdefault('identity_checks', {})[name] = float(same.max())
        entry['evaluation_wall_s'] = time.perf_counter()-begin
        results.append(entry)
        (args.out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(entry), flush=True)
    summary = {'status': 'complete', 'GPU_seconds': 0, 'evaluation_wall_s': time.perf_counter()-t0,
               'molecules': len(results), 'reference_cap': args.max_reference_conformers,
               'reference_scope': ('All conformers in the official reference ensemble; fixed candidate budget is not the published 2K convention.' if args.max_reference_conformers == 0 else 'First reference conformers in official file order; capped pilot metrics are not the published full-ensemble benchmark.'),
               'metric_source': 'Genentech/ringer f3ea676ec34512d25864088712a477f3c39cef91',
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'means': {}}
    for name, _, _ in functions:
        summary['means'][name] = {}
        for metric in ['cov_recall', 'cov_precision', 'cov_f1', 'mat_recall', 'mat_precision']:
            values = [r['metrics'][name][metric] for r in results]
            summary['means'][name][metric] = float(np.mean(values)) if all(v is not None for v in values) else None
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
