"""Compute statistics from generated single-clue predictions."""
from pathlib import Path
import csv
import json
import numpy as np


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    import argparse,os
    from hrm.paths import repository_root
    from .evaluate import summarize
    from hrm.analyse.dynamics.constraint_readout.run import finite
    p=argparse.ArgumentParser(description='Compute single-clue statistics from generated pairs')
    p.add_argument('--records',type=Path,default=Path(os.environ.get('HRM_ANALYSIS_RECORDS',str(repository_root()/'exp/generalization/records/single_clue'))))
    a=p.parse_args(argv)
    with np.load(a.records/'pairs.npz') as z:d=dict(z)
    result=summarize(d['pred_old'],d['pred_new'],d['solution_old'],d['solution_new'],d['cell'])
    (a.records/'statistics.json').write_text(json.dumps(finite(result),indent=2,allow_nan=False)+'\n')
    rows=[]
    for i in range(len(d['cell'])):
        rows.append(dict(id=str(d['ids'][i]) if 'ids' in d else i,changed_cell=int(d['cell'][i]),
                         changed_solution_cells=int((d['solution_old'][i]!=d['solution_new'][i]).sum()),
                         original_exact=bool(np.all(d['pred_old'][i]==d['solution_old'][i])),
                         edited_exact=bool(np.all(d['pred_new'][i]==d['solution_new'][i]))))
    write_csv(a.records/'per_pair_metrics.csv',rows)
    print(json.dumps(finite(result)))

if __name__ == '__main__':main()
