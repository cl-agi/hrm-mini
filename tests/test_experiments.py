import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hrm.experiments import experiments, experiment, compatibility_experiments, analysis_scope, records_directory, recipes, recipe_config
from hrm.paths import repository_root, repo_path, work_path, analysis_directory
from hrm.training.data import resolve_config, puzzle_directory, freeze_sampling_seed, freeze_training_seed


class ExperimentTests(unittest.TestCase):
    def test_sat_uses_its_task_specific_training_configuration(self):
        self.assertEqual(recipes('sat'), {})
        config=json.loads((repository_root()/'exp/sat/configs/inference/train.json').read_text())
        self.assertEqual(config['data']['name'], 'sat')
        self.assertEqual(config['arch']['vocab_size'], 4)
        for identifier in ('E42', 'E43'):
            self.assertEqual(recipes(identifier), {})

    def test_sat_intervention_has_one_default_records_scope(self):
        module='hrm.analyse.sat.constraint_intervention.summarize'
        scopes={e['records_root'] for e in compatibility_experiments().values()
                if e['group']=='error_readout' and module in e['modules']}
        self.assertEqual(scopes, {'exp/error_readout/records/sat_constraints'})

    def test_index_has_all_experiments_and_recipes(self):
        specs=experiments()
        self.assertEqual([x['id'] for x in specs], ['sudoku','sat','maze','cremp','ablations','dynamics','memory','error_readout','generalization'])
        self.assertEqual(set(compatibility_experiments()), {f'E{i:02d}' for i in range(1,52) if i != 48})
        recipes=json.loads((repository_root()/'exp/recipes.json').read_text())['recipes']
        self.assertEqual(set(recipes),{r for e in compatibility_experiments().values() for r in e['recipes']})
        for e in specs:
            for m in e['modules']:
                self.assertTrue((repository_root()/Path(*m.split('.'))).with_suffix('.py').exists(),m)

    def test_named_stage_and_legacy_identifier_route_to_same_records(self):
        module = 'hrm.analyse.dynamics.msd.analyze'
        legacy_specs, named_specs = compatibility_experiments(), experiments()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            target = root / 'exp/dynamics/records/msd'
            target.mkdir(parents=True)
            (target / 'sample.json').write_text('{}')
            with patch('hrm.experiments.repository_root', return_value=root), patch(
                'hrm.experiments.compatibility_experiments', return_value=legacy_specs
            ), patch('hrm.experiments.experiments', return_value=named_specs):
                self.assertEqual(records_directory('E24'), target)
                with analysis_scope(module, 'dynamics') as resolved:
                    self.assertEqual(resolved, target)
                    self.assertEqual(os.environ['HRM_ANALYSIS_RECORDS'], str(target))
                previous = root / 'exp/E24/records'
                previous.mkdir(parents=True)
                (previous / 'old.json').write_text('{}')
                self.assertEqual(records_directory('E24'), previous)
                self.assertEqual(records_directory('E24', target), target)

    def test_named_group_rejects_wrong_module_and_empty_records(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                with analysis_scope('hrm.analyse.dynamics.msd.analyze', 'sat', d):
                    pass
            with self.assertRaises(FileNotFoundError):
                with analysis_scope('hrm.analyse.dynamics.msd.analyze', 'dynamics', d):
                    pass
        self.assertIn('sudoku-hrm-3scale', recipes('ablations'))
        self.assertEqual(experiment('E24')['group'], 'dynamics')

    def test_fresh_recipe_does_not_invent_historical_seed(self):
        item=recipes()['sudoku-hrm-layers2-fresh-seed']
        before=copy.deepcopy(item)
        with self.assertRaisesRegex(ValueError,'Recipe requires resolution'):recipe_config(item)
        config=recipe_config(item,fresh_seed=True)
        self.assertNotIn('seeds',config)
        self.assertEqual(item,before)
        invalid=dict(item,unresolved_fields=['architecture'])
        with self.assertRaises(ValueError):recipe_config(invalid,fresh_seed=True)
        known={'status':'configuration_validated','config':{'seeds':[1,2,3]}}
        self.assertEqual(recipe_config(known)['seeds'],[1,2,3])
        self.assertNotIn('seeds',recipe_config(known,fresh_seed=True))

    def test_scalar_records_retain_denominator_and_selection_inputs(self):
        from hrm.training.records import record_evaluation
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evaluation.jsonl'
            for step,correct in [(10,1),(20,2)]:
                record_evaluation(p,seed=1794174000,step=step,split='test_hard',correct=correct,total=3,cycles=16,weights='EMA')
            rows=[json.loads(line) for line in p.read_text().splitlines()]
            self.assertEqual([r['accuracy'] for r in rows],[1/3,2/3])
            self.assertTrue(all(r['total']==3 and r['weights']=='EMA' for r in rows))
            with self.assertRaises(ValueError):record_evaluation(p,seed=1,step=1,split='test',correct=1,total=0,cycles=1,weights='raw')
            self.assertEqual(len(p.read_text().splitlines()),2)

    def test_training_summary_keeps_earliest_peak_and_last_separate(self):
        from hrm.training.records import record_evaluation
        from hrm.analyse.training_summary import summarize
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'run/evaluation.jsonl'
            for step,correct in [(10,2),(20,2),(30,1)]:
                record_evaluation(p,seed=1794174000,step=step,split='test',correct=correct,total=3,cycles=16,weights='EMA')
            result=summarize(root,'test')['runs'][0]
            self.assertEqual(result['peak']['optimizer_step'],10)
            self.assertEqual(result['last']['optimizer_step'],30)
            with self.assertRaises(FileNotFoundError):summarize(root,'absent')
            record_evaluation(p,seed=1794174000,step=30,split='test',correct=1,total=3,cycles=16,weights='EMA')
            with self.assertRaisesRegex(ValueError,'Duplicate'):summarize(root,'test')

    def test_summary_separates_budgets_and_rejects_mixed_stopping_rules(self):
        from hrm.training.records import record_evaluation
        from hrm.analyse.training_summary import summarize
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'run/evaluation.jsonl'
            for budget in (16,32):
                record_evaluation(p,seed=1794174000,step=10,split='test',correct=2,total=3,
                                  cycles=budget,weights='EMA',stable_cycles=2,mean_cycles=12.5)
            with self.assertRaisesRegex(ValueError,'budget'):summarize(root,'test')
            result=summarize(root,'test',16)['runs'][0]
            self.assertEqual(result['evaluations'],1)
            self.assertEqual(result['peak']['mean_inference_cycles'],12.5)
            record_evaluation(p,seed=1794174000,step=20,split='test',correct=2,total=3,
                              cycles=16,weights='EMA',stable_cycles=0)
            with self.assertRaisesRegex(ValueError,'stopping rule'):summarize(root,'test',16)

    def test_missing_records_fail_before_module_execution(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError): records_directory(path=d)
        with self.assertRaises(FileNotFoundError):records_directory('E47')

    def test_explicit_records_routes_inputs_and_restores_environment(self):
        before=os.environ.get('HRM_ANALYSIS_RECORDS');cwd=Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();(root/'sample.json').write_text('{}')
            with analysis_scope('hrm.analyse.sudoku.single_clue.prepare_statistics','E38',root):
                self.assertEqual(repo_path('data/example.csv'),root/'data/example.csv')
                self.assertEqual(work_path('summary.json'),root/'summary.json')
                self.assertEqual(repo_path('hrm/cli.py'),repository_root()/'hrm/cli.py')
                source=repository_root()/'hrm/analyse/sudoku/single_clue/prepare_statistics.py'
                self.assertEqual(analysis_directory(source),root/'hrm/analyse/sudoku/single_clue')
        self.assertEqual(os.environ.get('HRM_ANALYSIS_RECORDS'),before)
        self.assertEqual(Path.cwd(),cwd)

    def test_local_data_wins_and_does_not_download(self):
        config={'data':{'name':'hrm.tasks.sudoku'}}
        with tempfile.TemporaryDirectory() as d,patch('hrm.training.data.puzzle_directory',side_effect=AssertionError('network')):
            actual=resolve_config(config,{'variant':'unpublished'},dataset_path=d)
            self.assertEqual(actual['data']['dataset_name'],str(Path(d).resolve()))
            self.assertEqual(config,{'data':{'name':'hrm.tasks.sudoku'}})
            with self.assertRaises(FileNotFoundError):resolve_config(config,dataset_path=d+'/missing')

    def test_omitted_data_uses_hf_and_unpublished_never_substitutes(self):
        with patch('hrm.training.data.puzzle_directory',return_value=Path('/cache')) as p:
            result=resolve_config({'data':{'name':'hrm.tasks.sudoku'}})
            p.assert_called_once_with('sudoku','standard',30)
            self.assertEqual(result['data']['dataset_name'],'/cache')
        with self.assertRaisesRegex(FileNotFoundError,'not published'):puzzle_directory('sudoku','unpublished')

    def test_explicit_config_path_also_overrides_hf(self):
        with tempfile.TemporaryDirectory() as d,patch('hrm.training.data.puzzle_directory',side_effect=AssertionError('network')):
            result=resolve_config({'data':{'name':'hrm.tasks.sudoku','dataset_name':d}})
            self.assertEqual(result['data']['dataset_name'],str(Path(d).resolve()))

    def test_resampling_seed_is_frozen_and_shared_between_ranks(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {'RANK':'0'}), patch('time.time', return_value=1794174659):
                self.assertEqual(freeze_sampling_seed(d),1794174659)
            with patch.dict(os.environ, {'RANK':'1'}), patch('time.time', return_value=1794174660):
                self.assertEqual(freeze_sampling_seed(d),1794174659)
            with self.assertRaisesRegex(ValueError,'Saved data seed differs'):
                freeze_sampling_seed(d,1794174660)

    def test_sat_task_import_does_not_require_analysis_records(self):
        import subprocess,sys
        result=subprocess.run([sys.executable,'-c','from hrm.tasks.sat_flow import SATTask'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_default_training_seed_is_not_regenerated_by_worker_or_restart(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {'RANK':'0'}), patch('time.time',return_value=1794174000):
                self.assertEqual(freeze_training_seed(d),1794174000)
            with patch.dict(os.environ, {'RANK':'1'}), patch('time.time',return_value=1794174001):
                self.assertEqual(freeze_training_seed(d),1794174000)
            with patch.dict(os.environ, {'RANK':'0'}), patch('time.time',return_value=1794174002):
                self.assertEqual(freeze_training_seed(d),1794174000)

    def test_explicit_public_source_does_not_use_default_cohort(self):
        with patch('hrm.training.data.puzzle_directory',side_effect=AssertionError('wrong source')):
            for source in ('sapientinc/sudoku-extreme','hf://datasets/owner/dataset'):
                result=resolve_config({'data':{'name':'hrm.tasks.sudoku','dataset_name':source}})
                self.assertEqual(result['data']['dataset_name'],source)

if __name__=='__main__':unittest.main()
