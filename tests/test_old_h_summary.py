import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from hrm.analyse.sudoku.core.old_h_recovery import metrics
from hrm.analyse.dynamics.old_h_recovery.audit import statistics
from hrm.analyse.dynamics.old_h_recovery.summarize import pool,main
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha


class OldHSummaryTests(unittest.TestCase):
    def test_norms_and_pooling_use_ratio_of_sums(self):
        y=torch.tensor([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
        pred=y+torch.tensor([[1.,-1.],[2.,-2.],[3.,-3.],[4.,-4.]])
        mu=torch.zeros_like(y);blank=torch.tensor([True,False,True,False])
        got,arrays=metrics(pred,y,mu,blank,2,2,1794199000,10)
        other,independent=statistics(pred,y,mu,blank,2,2,1794199000,10)
        self.assertAlmostEqual(got['relative_l1'],10/18,6)
        self.assertAlmostEqual(got['relative_l2'],np.sqrt(60/204),6)
        for k in ('mae','target_abs','relative_l1','relative_l2'):self.assertAlmostEqual(got[k],other[k],6)
        for k in ('mae','target_abs'):np.testing.assert_allclose(arrays[k],independent[k])
        a=dict(mse=1.,baseline_mse=5.,target_energy=4.,mae=1.,target_abs=2.,cosine=.5,centered_cosine=.25)
        b=dict(a,mse=9.,target_energy=100.,mae=3.,target_abs=20.)
        combined=pool([a,b]);self.assertAlmostEqual(combined['relative_l1'],4/22)
        self.assertAlmostEqual(combined['relative_l2'],np.sqrt(10/104))
        self.assertNotAlmostEqual(combined['relative_l1'],(.5+.15)/2)
        zeros=dict.fromkeys(a,0.);self.assertIsNone(pool([zeros])['relative_l2']);self.assertIsNone(pool([zeros])['relative_l1'])

    def test_summary_rejects_unmatched_or_changed_conditions(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);records=root/'conditions'
            for i,case in enumerate(('continued_1','continued_2','continued_3','scratch')):
                folder=records/case
                for child in ('features','decoders','audit'):(folder/child).mkdir(parents=True)
                fp=folder/'features';dp=folder/'decoders'
                plan=dict(trained_hh_link=False,evaluation_hh_link=False,counts=[1,1,2],tokens=1,steps=[1],
                    board_ids=[0,1,2,3],positions=[[1]]*4,train_dataset_sha256='train',test_dataset_sha256='test',checkpoint_sha256=str(i))
                (fp/'run_config.json').write_text(json.dumps(plan));(fp/'features.pt').write_bytes(b'fixture')
                (fp/'manifest.json').write_text(json.dumps(dict(plan,status='complete',artifacts={'features.pt':sha(fp/'features.pt')})))
                config=dict(feature_manifest_sha256=sha(fp/'manifest.json'));(dp/'run_config.json').write_text(json.dumps(config))
                row=dict(step=1,probe='next_l_mlp',mse=1.,baseline_mse=2.,target_energy=4.,mae=.5,target_abs=1.,cosine=.9,centered_cosine=.8)
                (dp/'results.json').write_text(json.dumps([row]));(dp/'manifest.json').write_text(json.dumps(dict(config,status='complete',artifacts={'results.json':sha(dp/'results.json')})))
                (folder/'audit/audit.json').write_text(json.dumps(dict(status='complete',feature_manifest_sha256=sha(fp/'manifest.json'),decoder_manifest_sha256=sha(dp/'manifest.json'),checks=[dict(step=1,probe='next_l_mlp')])) )
            out=root/'summary';main(['--records',str(records),'--output-dir',str(out)])
            result=json.loads((out/'summary.json').read_text());self.assertEqual(result['metrics']['relative_l2'],.5);self.assertFalse(result['reference_sampling'])
            with self.assertRaises(FileExistsError):main(['--records',str(records),'--output-dir',str(out)])
            (records/'scratch/decoders/results.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'Changed artifact'):main(['--records',str(records),'--output-dir',str(root/'bad')])
