"""Separate reference process: official ring metrics, never all-heavy metrics."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
from rdkit import Chem
from hrm.analyse.cremp.classical.template_core import sha
pass
from hrm.analyse.cremp.classical.evaluate_conformers import subset, scores
from ringer.utils import evaluation

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference-manifest', type=Path, required=True)
    p.add_argument('--generation-results', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference-cap', type=int, default=0)
    p.add_argument('--cpus', type=int, default=8)
    a = p.parse_args()
    start = time.perf_counter()
    a.out.mkdir(parents=True, exist_ok=False)
    refs = json.loads(a.reference_manifest.read_text())
    refs = refs if isinstance(refs, list) else refs['entries']
    refs = {r['id']: r for r in refs}
    generated = json.loads(a.generation_results.read_text())
    rows = []
    functions = [('ring_rmsd', evaluation.compute_ring_rmsd_matrix, 0.1), ('ring_tfd', evaluation.compute_ring_tfd_matrix, 0.05)]
    for item in generated:
        assert item['sidechains_are_placeholders'] and item['allowed_metrics'] == ['ring_rmsd', 'ring_tfd']
        output = Path(item['output_molecule'])
        assert sha(output) == item['output_sha256']
        ref = refs[item['id']]
        path = Path(ref['reference_path'])
        if 'sha256' in ref:
            assert sha(path) == ref['sha256']
        reference = pickle.loads(path.read_bytes())
        reference = reference['rd_mol'] if isinstance(reference, dict) else reference
        reference = subset(reference, a.reference_cap)
        predicted = pickle.loads(output.read_bytes())
        count = predicted.GetNumConformers()
        row = dict(id=item['id'], num_reference=reference.GetNumConformers(), num_generated=count, requested_candidates=item['requested_candidates'], yield_fraction=count / item['requested_candidates'], unique_sampled_training_conformers=item['unique_sampled_training_conformers'], sidechains_are_placeholders=True, metrics={})
        for name, fn, threshold in functions:
            if count:
                matrix = fn(Chem.Mol(predicted), Chem.Mol(reference), ncpu=a.cpus)
                assert matrix.shape == (reference.GetNumConformers(), count) and np.isfinite(matrix).all()
                metric = scores(matrix, threshold)
                metric['cov_precision_per_requested'] = float(np.sum(matrix.min(axis=0) <= threshold) / item['requested_candidates'])
            else:
                matrix = np.empty((reference.GetNumConformers(), 0))
                metric = dict(threshold=threshold, cov_recall=0.0, cov_precision=0.0, cov_f1=0.0, cov_precision_per_requested=0.0, mat_recall=None, mat_precision=None)
            fname = output.stem + '_' + name + '.npz'
            np.savez_compressed(a.out / fname, distances=matrix)
            metric['distance_matrix_file'] = fname
            row['metrics'][name] = metric
        rows.append(row)
        (a.out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    summary = dict(status='complete', GPU_seconds=0, wall_s=time.perf_counter() - start, molecules=len(rows), reference_cap=a.reference_cap, reference_scope='All official conformers' if a.reference_cap == 0 else 'Capped engineering check', generation_results_sha256=sha(a.generation_results), reference_manifest_sha256=sha(a.reference_manifest), script_sha256=sha(__file__), allowed_metrics=['ring_rmsd', 'ring_tfd'], sidechains_are_placeholders=True, full_molecule_metrics_computed=False, means={})
    for name, _, _ in functions:
        summary['means'][name] = {}
        for key in ('cov_recall', 'cov_precision', 'cov_precision_per_requested', 'cov_f1', 'mat_recall', 'mat_precision'):
            values = [r['metrics'][name][key] for r in rows]
            summary['means'][name][key] = float(np.mean(values)) if all((v is not None for v in values)) else None
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
if __name__ == '__main__':
    main()
