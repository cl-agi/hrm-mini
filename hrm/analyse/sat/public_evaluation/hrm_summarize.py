from hrm.paths import analysis_directory
from pathlib import Path
import csv
import json
import sqlite3
import time

P = analysis_directory(__file__)
rows = []
queues = {}
for n in (512, 1024):
    d = P / f'n{n}'
    cfg = json.loads((d / 'config.json').read_text())
    db = sqlite3.connect(d / 'queue.sqlite')
    queues[n] = dict(db.execute('SELECT status,count(*) FROM tasks GROUP BY status'))
    results = [json.loads(p.read_text()) for p in (d / 'results').glob('task*.json')]
    for density, total in cfg['counts'].items():
        group = [r for r in results if r['density'] == density]
        for budget in cfg['budgets']:
            solved = sum(0 < r['first_valid_outer'] <= budget for r in group)
            evaluated = sum(r['first_valid_outer'] > 0 or r.get('assessed_outer', cfg['outer'] if r.get('complete') else 0) >= budget for r in group)
            rows.append(dict(variables=n, density=density, starts=64, outer=budget,
                             solved=solved, total=total, evaluated_formulas=evaluated,
                             fully_evaluated=evaluated == total,
                             observed_lower_bound_percent=100*solved/total, seed=cfg['seed']))
with (P / 'progress.tmp').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
(P / 'progress.tmp').replace(P / 'progress.csv')
if (P / 'stage.json').exists():
    stage = json.loads((P / 'stage.json').read_text())['outer']
    groups = [r for r in rows if r['outer'] == stage]
    report = dict(stage=stage, complete=all(r['fully_evaluated'] for r in groups),
                  evaluated_formulas=sum(r['evaluated_formulas'] for r in groups),
                  total=sum(r['total'] for r in groups), groups=groups, time=time.time())
    (P / 'stage_report.tmp').write_text(json.dumps(report, indent=2))
    (P / 'stage_report.tmp').replace(P / 'stage_report.json')
print(json.dumps(queues))
