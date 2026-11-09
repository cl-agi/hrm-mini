import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hrm_qwen.sft.data import read_split


class PublicDataTest(unittest.TestCase):
    def test_hf_default_and_leading_blanks(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'test.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['question','answer']);w.writerow(['.'+'1'*80,'1'*81])
            with patch('hrm.hub.puzzle_data',return_value=p) as resolve:
                rows=read_split(None,'test_hard')
            resolve.assert_called_once_with('sudoku',None,split='test_hard')
            self.assertEqual(rows[0][0],'0'+'1'*80)

    def test_missing_explicit_directory_does_not_download(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch('hrm.hub.dataset',side_effect=AssertionError('unexpected download')):
                with self.assertRaises(FileNotFoundError):read_split(temp,'train')

    def test_invalid_local_grid_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'train.csv').write_text('question,answer\n123,456\n')
            with self.assertRaises(ValueError):read_split(temp,'train')
