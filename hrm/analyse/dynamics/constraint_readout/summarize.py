"""Summarize saved constraint readouts and independently check held-out metrics."""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
from hrm.paths import analysis_directory, repository_root


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    default=(analysis_directory(__file__) if os.environ.get('HRM_ANALYSIS_RECORDS') else
             repository_root()/'exp/error_readout/records/sudoku_answers/hrm/analyse/dynamics/constraint_readout')
    p.add_argument('--records',type=Path,default=default);a=p.parse_args(argv);root=a.records
    result=json.loads((root/'results.json').read_text());start=sum(result['split'][:2]);rows=[]
    with np.load(root/'predictions.npz') as z, np.errstate(divide='ignore',invalid='ignore'):
        for target in ('soft','hard','hard_including_clues'):
            for probe in ('H','L','HL'):
                value=result[target][probe];true=z[target+'_true'][start:];pred=z[target+'_'+probe][start:]
                corr=float(np.corrcoef(true.ravel(),pred.ravel())[0,1])
                r2=1-((true-pred)**2).sum()/((true-true.mean())**2).sum()
                for actual,expected in ((corr,value['target']['r']),(r2,value['test_R2'])):
                    if expected is None:assert not np.isfinite(actual)
                    else:np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
                rows.append(dict(target=target,probe=probe,r=value['target']['r'],R2=value['test_R2'],
                    RMSE=value['test_RMSE'],r_low=value['target']['puzzle_bootstrap_95ci'][0],
                    r_high=value['target']['puzzle_bootstrap_95ci'][1],
                    within_time_r=value['target']['within_time_r'],
                    p=value['target']['within_time_puzzle_permutation_p_two_sided'],
                    next_L_r=value['next_L']['r']))
    with (root/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(probes=len(rows),test_puzzles=result['split'][2],
                         observations_per_puzzle=result['observations_per_puzzle'],metrics=str(root/'metrics.csv'))))
    return 0


if __name__=='__main__':raise SystemExit(main())
