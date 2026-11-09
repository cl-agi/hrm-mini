import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hrm.paths import dataset_path, checkpoint_config
from hrm.evaluation.config import load_config


class LocalEvaluationTests(unittest.TestCase):
    def test_explicit_dataset_does_not_need_archived_bindings(self):
        with tempfile.TemporaryDirectory() as d,patch('hrm.paths.repo_path',side_effect=AssertionError('archive lookup')):
            self.assertEqual(dataset_path(d),Path(d).resolve())

    @unittest.skipUnless(importlib.util.find_spec('datasets'),'datasets installation required')
    def test_csv_encoding_preserves_leading_zeros_and_long_digit_strings(self):
        from datasets import Features,Value
        from hrm.tasks.io import load_dataset
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);puzzle='0'*80+'1';answer='123456789'*9
            (root/'test.csv').write_text('question,answer,source\n'+puzzle+','+answer+',example\n')
            rows=load_dataset(str(root),split='test',features=Features({'question':Value('string'),'answer':Value('string')}),cache_dir=str(root/'cache'))
            self.assertEqual(rows[0],{'question':puzzle,'answer':answer})

    def test_adjacent_config_and_missing_config_diagnostic(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ck=root/'best.pt';ck.write_bytes(b'test')
            config=root/'model_config.json';config.write_text('{}')
            self.assertEqual(checkpoint_config(ck),config.resolve())
            config.unlink()
            with patch('hrm.paths.repo_path',return_value=root/'no-index.json'):
                with self.assertRaisesRegex(FileNotFoundError,'supply --config'):checkpoint_config(ck)

    def test_published_sat_json_preserves_numeric_configuration(self):
        import argparse
        from hrm.evaluation.published import add_arguments, run
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=root/'config.json'
            config.write_text(json.dumps({'arch':{'norm_eps':1e-6},'lr':1e-5}))
            parser=argparse.ArgumentParser();add_arguments(parser)
            args=parser.parse_args(['--task','sat','--config',str(config),'--output',str(root/'result.json')])
            with patch('hrm.analyse.sat.evaluate.evaluate',return_value={}) as evaluate, patch('hrm.hub.sat_eval_config',return_value={}):
                run(args)
            saved=evaluate.call_args.args[1]
            self.assertIsInstance(saved['arch']['norm_eps'],float)
            self.assertIsInstance(saved['lr'],float)

    def test_explicit_data_overrides_stale_configuration_before_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'inputs';data.mkdir()
            config=root/'config.json'
            value={'arch':{'name':'hrm.models.hrm@HRM','norm_eps':1e-6},'data':{'name':'hrm.tasks.sudoku','dataset_name':'/missing/historical/data'},'cycles_per_data':2,'local_batch_size':2}
            config.write_text(json.dumps(value))
            result=load_config(root/'best.pt',config,dataset=data)
            self.assertEqual(result.data.model_dump()['dataset_name'],str(data.resolve()))
            self.assertIsInstance(result.model_config_dict({})['norm_eps'],float)
            self.assertEqual(result.model_config_dict({})['norm_eps'],1e-6)
            with self.assertRaises(FileNotFoundError):load_config(root/'best.pt',config)
            self.assertEqual(json.loads(config.read_text()),value)


if __name__=='__main__':unittest.main()
