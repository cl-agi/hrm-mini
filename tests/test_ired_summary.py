import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec('hrm_ired'), 'optional IRED package')
class IredSummaryTests(unittest.TestCase):
    def test_derived_counts_and_mismatch_rejection(self):
        import numpy as np
        import torch
        from hrm.analyse.baselines.ired.summarize_diagnostics import summarize
        class Reference:
            def __len__(self):return 2
            def __getitem__(self,i):return None,torch.nn.functional.one_hot(torch.zeros(81,dtype=torch.long),9).flatten(),None
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'variants.json'
            manifest.write_text(json.dumps(dict(baseline='base',variants={'base':'base','other':'other'})))
            for name,correct in [('base',1),('other',2)]:
                folder=root/name;folder.mkdir()
                (folder/'config.json').write_text(json.dumps(dict(splits=['rrn'])))
                (folder/'rrn.json').write_text(json.dumps(dict(n=2,seed=42,exact_match=correct/2,strict_valid_and_clues=0,clues_preserved=0)))
                np.save(folder/'rrn.indices.npy',np.array([0,1]))
                pred=np.ones((2,81),dtype=np.int64)
                if correct==1:pred[1]=2
                np.save(folder/'rrn.predictions.npy',pred)
            with patch('hrm_ired.sat_dataset.SudokuRRNDataset',return_value=Reference()):
                result=summarize(manifest)
                paired=result['results']['other']['paired_rrn_vs_baseline']
                self.assertEqual(paired['newly_solved'],1)
                self.assertEqual(paired['delta_percentage_points'],50)
                self.assertNotIn('conclusions',result)
                np.save(root/'other/rrn.indices.npy',np.array([1,0]))
                with self.assertRaisesRegex(ValueError,'order differs'):summarize(manifest)
                np.save(root/'other/rrn.indices.npy',np.array([0,1]))
                metric=root/'other/rrn.json';data=json.loads(metric.read_text());data['exact_match']=0;metric.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,'disagrees'):summarize(manifest)
            (root/'other/rrn.predictions.npy').unlink()
            with patch('hrm_ired.sat_dataset.SudokuRRNDataset',side_effect=AssertionError('must fail before reference loading')):
                with self.assertRaises(FileNotFoundError):summarize(manifest)
