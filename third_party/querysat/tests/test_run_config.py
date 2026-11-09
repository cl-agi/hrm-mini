import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from hrm_querysat.run_config import prepare


def config(root, **overrides):
    values = dict(train_dir=str(root), restore=None, seed=None, data_dir=None,
                  label='', force_data_gen=False, model='querysat', task='ksat',
                  learning_rate=.0002, warmup=0., train_steps=10000, ckpt_count=3, eager=False)
    values.update(overrides)
    return SimpleNamespace(**values)


class RunConfigTest(unittest.TestCase):
    def test_preparation_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            first = config(directory)
            with patch('hrm_querysat.run_config.time.time', return_value=1794183000):
                saved = prepare(first)
            self.assertEqual(saved['seed'], 1794183000)
            self.assertEqual(Path(saved['data_dir']), Path(first.train_dir).resolve() / 'data')
            self.assertEqual(json.loads((Path(first.train_dir)/'resolved_config.json').read_text()),saved)
            restored = config(directory, restore=first.train_dir, learning_rate=99)
            self.assertEqual(prepare(restored), saved)
            self.assertEqual(restored.learning_rate, .0002)
            with self.assertRaises(ValueError):
                prepare(config(directory, restore=first.train_dir, seed=42))
            with self.assertRaises(ValueError):
                prepare(config(directory, restore=first.train_dir, force_data_gen=True))
            with self.assertRaises(FileExistsError):
                prepare(config(directory, seed=1794183000))

    def test_missing_restore_does_not_start_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                prepare(config(directory, restore=str(Path(directory)/'missing')))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_explicit_missing_data_fails_before_creating_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                prepare(config(directory, data_dir=str(Path(directory)/'missing')))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_evaluation_requires_weights(self):
        import tensorflow as tf
        from hrm_querysat.main import prepare_checkpoints
        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(train_dir=directory, ckpt_count=3, train=False)
            with patch('hrm_querysat.main.Config', settings):
                with self.assertRaisesRegex(FileNotFoundError, 'evaluation requires'):
                    prepare_checkpoints(tf.Module(), tf.keras.optimizers.legacy.Adam())
