import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from hrm.analyse.dynamics.msd.analyze import main, load_curves, analyse, summarize, summarize_curves, SERIES, EVAL_L

LAGS=[1,2,3,4,5,6,7,8,9,10,13,14,15,17,19,20,22,26,27,28,30,36,38,44,45,47,53,69,74,76,103,106,130,143,161,223,246,376,383,575,657,1128,1938,3327]


def write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class MSDTests(unittest.TestCase):
    def test_normalize_before_seed_average(self):
        rows=[]
        # Different baseline scales: ratio of averages (301/101) is wrong;
        # averaging per-seed normalized values gives (1+3)/2 = 2.
        for seed,base,value in [(1794198000,1.,1.),(1794198001,100.,300.)]:
            rows.append(dict(condition='hrm',state='h',seed=seed,segment=1,lag=7,
                window_start=0,window_end=48,origins=41,msd=value,msd_at_6=base,normalized_msd=value/base))
        summary=summarize_curves(rows)[0]
        self.assertEqual(summary['normalized_msd_mean'],2.)
        self.assertAlmostEqual(summary['normalized_msd_sd'],np.sqrt(2))
        self.assertIsNone(summarize_curves(rows[:1])[0]['normalized_msd_sd'])
        for row in rows:row['normalized_msd']=None
        summary=summarize_curves(rows)[0]
        self.assertEqual(summary['defined_runs'],0)
        self.assertIsNone(summary['normalized_msd_mean'])

    def test_shared_windows_and_training_seed_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);metadata=[];curves=[];bounds=[0,48,192,768,4096]
            for condition,states in SERIES.items():
                for j in range(3):
                    seed=1794197000+j
                    metadata.append(dict(condition=condition,seed=seed,samples=256,eval_l=EVAL_L[condition],best_epoch=19,actual_l_updates=4096,segment_boundaries_l_updates=json.dumps(bounds),lag_l_updates=json.dumps(LAGS)))
                    for state in states:
                        beta=.2+j*.1+(.3 if state=='h' else 0)
                        for segment in range(1,5):
                            size=bounds[segment]-bounds[segment-1]
                            for lag in LAGS:
                                if lag<size:curves.append(dict(condition=condition,state=state,seed=seed,segment=segment,lag_l_updates=lag,origins=size-lag,mean=2.5*lag**beta,eval_l=EVAL_L[condition],puzzles=256,status='defined'))
            write(root/'rollout_metadata.csv',metadata);write(root/'msd_curves.csv',curves)
            main(['--records',str(root),'--output-dir',str(root/'stats')])
            report=json.loads((root/'stats/analysis_manifest.json').read_text())
            self.assertEqual(report['outputs']['msd_curve_summary.csv']['rows'],1008)
            self.assertEqual([report['outputs'][key]['rows'] for key in ('msd_curves.csv','beta_per_seed.csv','beta_summary.csv','delta_per_seed.csv','delta_summary.csv')],[3024,273,91,117,39])
            _,groups,_=load_curves(root);fits=analyse(groups);summary,delta,ds=summarize(fits)
            for row in fits:self.assertAlmostEqual(row['beta'],.2+(row['seed']-1794197000)*.1+(.3 if row['state']=='h' else 0))
            for row in summary:self.assertAlmostEqual(row['beta_sd'],.1)
            for row in delta:self.assertAlmostEqual(row['delta_beta'],.3)
            with self.assertRaises(FileExistsError):main(['--records',str(root),'--output-dir',str(root/'stats')])
            # Undefined constant-zero state has no logarithmic exponent, not beta=0.
            for row in curves:
                if row['condition']=='hrm' and row['state']=='h':row['mean']=0
            write(root/'msd_curves.csv',curves);_,groups,_=load_curves(root)
            undefined=[row for row in analyse(groups) if row['condition']=='hrm' and row['state']=='h']
            self.assertTrue(all(row['beta'] is None and row['n_lags']==0 for row in undefined))
            write(root/'msd_curves.csv',curves[:-1])
            with self.assertRaisesRegex(ValueError,'Missing'):load_curves(root)


if __name__=='__main__':unittest.main()
