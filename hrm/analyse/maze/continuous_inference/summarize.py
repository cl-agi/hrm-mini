"""Audit exact-once shard coverage before reporting a full Maze cohort."""
import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import summarize


def combine(directories):
    plans, traces, indices, ids = [], [], [], []
    for directory in map(Path, directories):
        plan = json.loads((directory/'plan.json').read_text())
        receipt = json.loads((directory/'summary.json').read_text())
        if receipt.get('status') != 'complete' or not receipt.get('final_cpu_audit'):
            raise ValueError(f'Incomplete evaluation: {directory}')
        if any(receipt.get(k) != v for k, v in plan.items()):
            raise ValueError('Plan/receipt mismatch')
        local_indices, local_ids, local_trace = [], [], []
        for i in range((len(plan['indices']) + plan['batch_size'] - 1)//plan['batch_size']):
            with np.load(directory/f'batch_{i}.npz', allow_pickle=False) as f:
                local_indices.extend(f['indices'].tolist()); local_ids.extend(f['ids'].tolist())
                local_trace.append(f['metrics'])
        trace = np.concatenate(local_trace)
        if local_indices != plan['indices'] or local_ids != plan['ids']:
            raise ValueError('Batch order/identity mismatch')
        if trace.shape != (len(local_indices), plan['cycles'], 2) or trace.dtype != np.bool_:
            raise ValueError('Invalid metric array')
        if any(receipt.get(k) != v for k, v in summarize(trace).items()):
            raise ValueError('Saved counts do not match metrics')
        if plans and any(plan[k] != plans[0][k] for k in
                         ('kind', 'checkpoint_sha256', 'input_sha256', 'cycles', 'total_count', 'shards')):
            raise ValueError('Mixed checkpoints, inputs or inference budgets')
        expected = list(range(plan['total_count']))[plan['shard']::plan['shards']]
        if plan['indices'] != expected:
            raise ValueError('Invalid shard partition')
        plans.append(plan); traces.append(trace); indices.extend(local_indices); ids.extend(local_ids)
    if not plans:
        raise ValueError('No evaluation directories')
    if sorted(indices) != list(range(plans[0]['total_count'])) or len(set(ids)) != len(ids):
        raise ValueError('Missing or duplicate examples')
    trace = np.concatenate(traces)[np.argsort(indices)]
    return {k: plans[0][k] for k in ('kind', 'checkpoint_sha256', 'input_sha256')} | summarize(trace)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories', nargs='+')
    a = p.parse_args()
    print(json.dumps(combine(a.directories), indent=2))


if __name__ == '__main__':
    main()
