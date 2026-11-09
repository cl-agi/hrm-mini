"""Puzzle-cluster bootstrap of Qwen correctness records."""
from hrm.paths import analysis_directory
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

root = analysis_directory(__file__)
source = root / 'qwen_correctness.csv'
rows = list(csv.DictReader(source.open()))
conditions = ['Qwen Instruct', 'Qwen teacher-CoT SFT']
rates = []
for condition in conditions:
    subset = [r for r in rows if r['condition'] == condition]
    keys = [(int(r['puzzle_index']), int(r['generation_position'])) for r in subset]
    assert len(keys) == len(set(keys)) == 1024
    assert set(keys) == {(i, j) for i in range(256) for j in range(1, 5)}
    values = {(int(r['puzzle_index']), int(r['generation_position'])): int(r['correct']) for r in subset}
    assert set(values.values()) <= {0, 1}
    rates.append(np.array([sum(values[i, j] for j in range(1, 5))/4 for i in range(256)]))
# Resample puzzle clusters jointly, retaining all four outputs and both conditions.
seed = 20261025
weights = np.random.default_rng(seed).multinomial(256, np.full(256, 1/256), size=10000)/256
result = {'n_puzzles': 256, 'outputs_per_puzzle_per_condition': 4, 'bootstrap_replicates': 10000,
          'analysis_seed': seed, 'method': 'paired puzzle-cluster percentile bootstrap',
          'source': str(source.relative_to(root.parent.parent)), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'conditions': {}}
def stats(v):
    lo, hi = np.quantile(weights @ v, [.025, .975])*100
    return {'mean_percent': float(v.mean()*100), 'ci95_low_percent': float(lo), 'ci95_high_percent': float(hi)}
for condition, v in zip(conditions, rates):
    result['conditions'][condition] = stats(v)
result['paired_SFT_minus_Instruct_percentage_points'] = stats(rates[1]-rates[0])
(root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
with (root/'per_puzzle.csv').open('w') as f:
    w=csv.writer(f, lineterminator='\n');w.writerow(['puzzle_index','instruct_mean_correct','sft_mean_correct','paired_difference'])
    for i in range(256): w.writerow([i,rates[0][i],rates[1][i],rates[1][i]-rates[0][i]])
print(json.dumps(result, indent=2))
