import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec('hrm_neurosat'), 'optional NeuroSAT package')
class NeuroSATPretrainedTests(unittest.TestCase):
    def test_explicit_checkpoint_never_downloads_and_cache_is_verified(self):
        from hrm_neurosat.pretrained import checkpoint
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.ckpt';path.write_bytes(b'fixture')
            with patch('urllib.request.urlopen',side_effect=AssertionError('must not download')):
                self.assertEqual(checkpoint(path),path)
                with self.assertRaises(FileNotFoundError):checkpoint(path.with_name('missing.ckpt'))
                with patch('hrm.hub.cache_path',return_value=path):
                    with self.assertRaisesRegex(ValueError,'changed'):checkpoint()
    def test_invalid_iteration_budget_fails_before_download(self):
        from hrm_neurosat.pretrained import load_model
        with patch('hrm_neurosat.pretrained.checkpoint',side_effect=AssertionError('must not download')):
            with self.assertRaisesRegex(ValueError,'positive'):load_model(iterations=0)

    def test_analysis_loader_uses_explicit_checkpoint_path(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from hrm.analyse.sat.graph_coloring import neurosat
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'chosen.ckpt';path.write_bytes(b'checkpoint')
            options=SimpleNamespace(model='neurosat',graph='lcg',init_emb='learned',task='assignment')
            saved={'hyper_parameters':{'args':options},'state_dict':{},'epoch':1,'global_step':2}
            model=MagicMock();model.eval.return_value.requires_grad_.return_value.to.return_value=model
            model.parameters.return_value=[]
            with patch.object(neurosat,'sha256',return_value=neurosat.CHECKPOINT_SHA256) as digest, \
                 patch.object(neurosat.torch,'load',return_value=saved) as load, \
                 patch('hrm_neurosat.gnn.GNN',return_value=model):
                actual,metadata=neurosat.load_model('cpu',path)
            self.assertIs(actual,model);self.assertEqual(metadata['global_step'],2)
            digest.assert_called_once_with(path)
            load.assert_called_once_with(path,map_location='cpu',weights_only=False)
