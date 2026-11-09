import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile
import json

@unittest.skipUnless(importlib.util.find_spec('hrm_ired'), 'optional IRED package')
class IredPublicDataTests(unittest.TestCase):
    def test_explicit_path_and_cache_integrity(self):
        from hrm_ired.public_data import directory, digest, SOURCES
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            for name in ('features.pt','labels.pt'):(root/name).write_bytes(name.encode())
            with patch('urllib.request.urlopen',side_effect=AssertionError('unexpected download')):
                self.assertEqual(directory('sudoku',root),root)
                with self.assertRaises(FileNotFoundError):directory('sudoku',root/'missing')
                (root/'source.json').write_text(json.dumps(dict(archive_sha256=SOURCES['sudoku'][1],files={n:digest(root/n) for n in SOURCES['sudoku'][3]})))
                with patch('hrm.hub.cache_path',return_value=root/'placeholder'):
                    self.assertEqual(directory('sudoku'),root)
                    (root/'features.pt').write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError,'cache changed'):directory('sudoku')

    def test_rrn_leading_zero_and_satnet_partition(self):
        import torch
        from hrm_ired.sat_dataset import SudokuDataset, SudokuRRNDataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            features=torch.zeros(10,9,9,9);labels=features.clone();labels[:,:,:,0]=1
            torch.save(features,root/'features.pt');torch.save(labels,root/'labels.pt')
            self.assertEqual(len(SudokuDataset('sudoku','train',root)),9)
            self.assertEqual(len(SudokuDataset('sudoku','val',root)),1)
            (root/'test.csv').write_text('0'*81+','+'1'*81+'\n')
            inp,label,mask=SudokuRRNDataset('sudoku-rrn','test',root)[0]
            self.assertEqual(inp.numel(),729)
            self.assertTrue((inp==-1).all())
            self.assertEqual(mask.sum(),0)
