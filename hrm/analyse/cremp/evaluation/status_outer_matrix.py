"""Compact read-only progress with paired-noise assertions."""
from hrm.paths import repo_path, work_path
from pathlib import Path
import json
from pathlib import Path
b = Path(str(work_path('ai4s_cremp')))
for seed in (1, 2, 3):
    rows = []
    hashes = []
    for family in ('hrm', 'rt'):
        for mode in ('retain', 'reset'):
            name = f'outer_r2_{family}_{mode}_seed{seed}'
            p = b / 'runs' / name
            if (p / 'pipeline_error.json').exists():
                raise RuntimeError((p / 'pipeline_error.json').read_text())
            if not (p / 'metrics.json').exists():
                rows.append([family, mode, 'pending'])
                continue
            metrics = json.loads((p / 'metrics.json').read_text())
            r = metrics[-1]
            rows.append([family, mode, r['step'], round(r.get('val_loss', 0), 6), (b / 'inference' / name / 'pipeline_summary.json').exists()])
            hashes.append({x['step']: x['corruption_sha256'] for x in metrics if 'corruption_sha256' in x})
    common = set.intersection(*(set(x) for x in hashes)) if hashes else set()
    assert all((len({x[k] for x in hashes}) == 1 for k in common))
    print(json.dumps(dict(seed=seed, arms=rows, paired_arms=len(hashes), paired_noise_points=len(common))))
