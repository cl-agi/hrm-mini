import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hrm.analyse.maze.continuous_inference import prepare as module
from hrm.analyse.maze.continuous_inference.data import make_model


class MazeContinuousDataTests(unittest.TestCase):
    def test_partition_complete_disjoint_and_frozen(self):
        def encode(path):
            shift=0 if path.name=='train.csv' else 1000
            x=np.arange(shift,shift+1000).reshape(-1,1)
            return dict(x=x,y=x.copy(),ids=np.arange(1000),ds=x.copy(),dg=x.copy())
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir()
            for s in ('train','test'):(source/(s+'.csv')).write_text(s)
            with patch.object(module,'encode',side_effect=encode),patch('time.time',return_value=1794192000):
                output=module.prepare(root/'prepared',source)
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['seed'],1794192000)
            self.assertEqual(manifest['splits'],dict(train=1000,val=200,test=800))
            with np.load(output/'maze_val.npz') as v,np.load(output/'maze_test.npz') as t:
                self.assertFalse(set(v['ids'])&set(t['ids']))
                self.assertEqual(set(v['ids'])|set(t['ids']),set(range(1000)))
            self.assertEqual(module.prepare(output,source),output)
            (source/'train.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Local source changed'):module.prepare(output,source)
            (source/'train.csv').write_text('train')
            with self.assertRaisesRegex(ValueError,'seed differs'):module.prepare(output,source,1)
            (output/'maze_test.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Prepared data changed'):module.prepare(output,source)

    def test_trm_resolves_to_shared_ablations_not_imported_plain_hrm(self):
        from hrm.models.ablations import HRM
        small=dict(seq_len=901,vocab_size=6,hidden_size=8,intermediate_size=16,head_dim=4,num_layers=1,
                   H_cycles=1,L_cycles=1,bptt=True,norm_eps=1e-6,rope_theta=10000.,forward_dtype='float32',dual_module=False,hh_link=True,sym_io=False)
        with patch('hrm.analyse.maze.continuous_inference.data.model_config',return_value=small):
            model=make_model('maze','trm',device='cpu')
        self.assertIsInstance(model,HRM)
        self.assertIs(model.H_level,model.L_level)
        import torch
        from hrm.training.steps import forward
        carry,logits=forward(model,model.initial_carry,torch.zeros(1,901,dtype=torch.long),attn_mask=None)
        self.assertEqual(tuple(logits.shape),(1,901,6))
        self.assertTrue(torch.isfinite(logits).all())

if __name__=='__main__':unittest.main()
