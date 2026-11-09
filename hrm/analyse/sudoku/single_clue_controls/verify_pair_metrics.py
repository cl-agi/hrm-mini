"""Verify the statistics against the generated per-pair predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from hrm.paths import repository_root
from hrm.analyse.sudoku.single_clue.evaluate import summarize
from hrm.analyse.dynamics.constraint_readout.run import finite


def verify(records=None):
    root=Path(records) if records else repository_root()/'exp/generalization/records/single_clue'
    with np.load(root/'pairs.npz') as z:d=dict(z)
    actual=finite(summarize(d['pred_old'],d['pred_new'],d['solution_old'],d['solution_new'],d['cell']))
    expected=json.loads((root/'statistics.json').read_text())
    if actual!=expected:raise ValueError('Statistics disagree with per-pair predictions')
    return dict(passed=True,num_pairs=actual['num_pairs'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path);a=p.parse_args()
    print(json.dumps(verify(a.records)))

if __name__=='__main__':main()
