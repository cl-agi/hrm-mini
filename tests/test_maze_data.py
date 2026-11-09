import unittest
from unittest.mock import patch
from hrm.training.maze_data import load_rows


class MazeDataTests(unittest.TestCase):
    def rows(self):
        return [dict(id=f'{size}-{i}',maze=[' '*size]*size,solution=[' '*size]*size)
                for size in (15,20,25,30) for i in range(250)]

    def test_mixed_training_cohort_is_not_resampled_or_truncated(self):
        rows=self.rows()
        with patch('hrm.hub.dataset',return_value=rows) as fetch:
            self.assertIs(load_rows(),rows)
            fetch.assert_called_once_with('maze','train')
        with patch('hrm.hub.dataset',return_value=rows[:-1]):
            with self.assertRaisesRegex(ValueError,'250'):load_rows()

    def test_duplicate_ids_and_malformed_boards_fail(self):
        rows=self.rows();rows[-1]['id']=rows[0]['id']
        with patch('hrm.hub.dataset',return_value=rows):
            with self.assertRaisesRegex(ValueError,'unique'):load_rows()
        rows=self.rows();rows[0]['maze'][0]='?'
        with patch('hrm.hub.dataset',return_value=rows):
            with self.assertRaisesRegex(ValueError,'Invalid'):load_rows()

    def test_missing_explicit_path_never_falls_back_to_hf(self):
        try:import pyarrow
        except ImportError:self.skipTest('optional data dependency unavailable')
        with patch('hrm.hub.dataset',side_effect=AssertionError('unexpected download')):
            with self.assertRaises(FileNotFoundError):load_rows(path='/nonexistent-maze-input.parquet')


if __name__=='__main__':unittest.main()
