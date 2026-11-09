import importlib.util
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import csv

@unittest.skipUnless(importlib.util.find_spec('hrm_ired'), 'optional IRED package')
class IredDataTests(unittest.TestCase):
    def test_augmentation_preserves_solution_and_clues(self):
        import numpy as np
        from hrm.analyse.baselines.ired.train_extreme_aug import Train1k, augment_pair
        solution=np.array([[(r*3+r//3+c)%9+1 for c in range(9)] for r in range(9)])
        puzzle=solution.copy();puzzle.flat[::3]=0
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'train.csv'
            with path.open('w') as stream:
                writer=csv.writer(stream);writer.writerow(['question','answer'])
                writer.writerows([[''.join(map(str,puzzle.flat)),''.join(map(str,solution.flat))]]*1000)
            a=Train1k(True,123,path);b=Train1k(True,123,path)
            for _ in range(3):
                x,y,mask=a[0]; x2,y2,mask2=b[0]
                self.assertTrue((x==x2).all() and (y==y2).all() and (mask==mask2).all())
                self.assertEqual(tuple(x.shape),(729,))
                self.assertTrue((x[mask.bool()]==y[mask.bool()]).all())
            x,y=augment_pair(puzzle,solution,np.random.default_rng(124))
            self.assertTrue(((x==0)|(x==y)).all())
            self.assertTrue((np.sort(y,axis=0)==np.arange(1,10)[:,None]).all())
            self.assertTrue((np.sort(y,axis=1)==np.arange(1,10)).all())
            with patch('hrm.hub.dataset',side_effect=AssertionError('must not fetch')):
                with self.assertRaises(FileNotFoundError):Train1k(False,123,Path(folder)/'missing.csv')

    def test_evaluation_uses_official_100_problem_split(self):
        import numpy as np
        from hrm.analyse.baselines.ired.eval_extreme import data
        solution=''.join(str((r*3+r//3+c)%9+1) for r in range(9) for c in range(9))
        with tempfile.TemporaryDirectory() as folder:
            for name,n in [('test_hard.csv',20000),('test_sudoku_bench.csv',100)]:
                with (Path(folder)/name).open('w') as stream:
                    writer=csv.writer(stream);writer.writerow(['question','answer']);writer.writerows([['0'*81,solution]]*n)
            with patch('huggingface_hub.hf_hub_download',side_effect=AssertionError('explicit path must not fetch')):
                ds,paths=data(folder)
            self.assertEqual(len(ds),20100)
            self.assertEqual(ds.x.dtype,np.int64)
            x,y,mask=ds[20000]
            self.assertEqual(tuple(y.shape),(729,))
            self.assertEqual(mask.sum(),0)
