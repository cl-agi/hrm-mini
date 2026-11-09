import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from hrm.paths import repository_root
from hrm.training.data_scaling_records import verify_e10_training,ordered_dataset_identity


def fixture(root):
    root.mkdir(exist_ok=True)
    config=json.loads((repository_root()/'exp/sudoku/configs/stable_stopping/hrm.json').read_text())
    config.update(seeds=[1794191000],output_dir=str(root));config.pop('training_policy')
    records=[]
    for i in range(1,21):
        step=i*4160;correct=16000 if i>=19 else 15000
        ck=root/f'step_{step}.pt';ck.write_bytes(f'fixture weights {step}'.encode())
        records.append(dict(evaluation=i,seed=config['seeds'][0],world_size=8,global_batch_size=768,
            interval_trained_examples=260*768,trained_examples=i*260*768,optimizer_steps=step,
            optimizer_example_exposures=step*768,cycles_per_data=16,checkpoint=str(ck),
            checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
            eval={'test':dict(correct=correct,total=20000,exact_match=correct/20000,mean_cycles=16,max_cycles=16,stable_cycles=16)}))
    summary=dict(status='complete',seed=config['seeds'][0],total_steps=83200,num_evals=20,optimizer_steps=83200,
                 trained_examples=5200*768,optimizer_example_exposures=83200*768,best_eval=records[-1],final_eval=records[-1])
    provenance=dict(seed=config['seeds'][0],world_size=8,data=config['data'],train=dict(samples=1000,ordered_sha256='train-hash'),test=dict(samples=20000,ordered_sha256='test-hash'))
    for name,obj in [('model_config.json',config),('training_summary.json',summary),('input_provenance.json',provenance)]:
        (root/name).write_text(json.dumps(obj))
    (root/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    return dict(config=config,split='test',samples=20000,dataset_order_sha256='test-hash',checkpoint_sha256=records[-1]['checkpoint_sha256'])


class E10SelectionTests(unittest.TestCase):
    def test_latest_tie_and_portable_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=fixture(root)
            result=verify_e10_training(root,plan,1794191000)
            self.assertEqual(result['optimizer_step'],83200)
            self.assertEqual(len(result['evidence_hashes']),4)

    def test_missing_schedule_wrong_cohort_and_wrong_weights_fail(self):
        for mode in ['missing','cohort','weights','earliest','architecture','step_count']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d);plan=fixture(root)
                if mode=='missing':
                    p=root/'metrics.jsonl';p.write_text('\n'.join(p.read_text().splitlines()[1:])+'\n')
                elif mode=='cohort':plan['dataset_order_sha256']='other'
                elif mode=='weights':(root/'step_83200.pt').write_bytes(b'changed')
                elif mode=='architecture':plan['config']['transformer']['hidden_size']=256
                else:
                    p=root/'training_summary.json';s=json.loads(p.read_text())
                    if mode=='earliest':s['best_eval']=json.loads((root/'metrics.jsonl').read_text().splitlines()[-2])
                    else:s['optimizer_steps']=83199
                    p.write_text(json.dumps(s))
                with self.assertRaises(ValueError):verify_e10_training(root,plan,1794191000)

    def test_native_evidence_through_curve_summary(self):
        import numpy as np
        from hrm.analyse.sudoku.stable16.summarize import summarize
        from hrm.analyse.sudoku.stable16.evaluate_distributed import validate_complete_results
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);training=root/'training';plan=fixture(training)
            directory=root/'curve';directory.mkdir()
            plan.update(cycles=[16,32],stable_cycles=16,checkpoint=str(training/'step_83200.pt'),world_size=1)
            (directory/'run_config.json').write_text(json.dumps(plan))
            combined=dict(complete=True,cycle_budgets=[16,32],shared_inference=True,checkpoint=plan['checkpoint'],stable_cycles=16,results={})
            for limit in [16,32]:
                folder=directory/f'cycles_{limit}';folder.mkdir()
                shard=dict(indices=np.arange(20000),correctness=np.arange(20000)<16000,cycles=np.full(20000,min(limit,17)))
                np.savez(folder/'rank_00000.npz',**shard)
                metrics=validate_complete_results([shard],20000,limit,16)
                row=dict(complete=True,input_provenance=plan,checkpoint=plan['checkpoint'],stable_cycles=16,
                    world_size=1,requested_max_cycles=limit,shared_inference_max_cycles=32,**metrics)
                (folder/'summary.json').write_text(json.dumps(row))
                combined['results'][str(limit)]=dict(summary=f'cycles_{limit}/summary.json',**metrics)
            (directory/'summary.json').write_text(json.dumps(combined))
            manifest=root/'curve_runs.json';entry=dict(family='HRM',training_seed=1794191000,directory='curve',training_directory='training')
            manifest.write_text(json.dumps(dict(runs=[entry])))
            report=summarize(manifest,expected_runs=1,require_e10_training=True)
            run=report['groups']['HRM']['runs'][0]
            self.assertTrue(run['e10_training_verified'])
            self.assertEqual(run['training']['optimizer_step'],83200)
            self.assertFalse(run['scalar_selection_verified'])
            entry.pop('training_directory');manifest.write_text(json.dumps(dict(runs=[entry])))
            with self.assertRaisesRegex(ValueError,'Missing native'):
                summarize(manifest,expected_runs=1,require_e10_training=True)

    def test_identity_uses_curve_hash_encoding(self):
        rows=[dict(question='001',answer='123'),dict(question='010',answer='312')]
        expected=hashlib.sha256(b'["001","123"]\n["010","312"]\n').hexdigest()
        self.assertEqual(ordered_dataset_identity(rows),dict(samples=2,ordered_sha256=expected))

if __name__=='__main__':unittest.main()
