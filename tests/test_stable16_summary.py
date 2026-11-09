import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from hrm.analyse.sudoku.stable16.summarize import summarize
from hrm.analyse.sudoku.stable16.evaluate_distributed import validate_complete_results


def write_run(root, seed, correct):
    root.mkdir()
    plan=dict(cycles=[16,32],stable_cycles=16,checkpoint=f'model-{seed}.pt',checkpoint_sha256=str(seed)*64,
        world_size=1,samples=2,dataset_order_sha256='same',split='test',config={'arch':{'name':'hrm@HRM'}})
    (root/'run_config.json').write_text(json.dumps(plan))
    combined=dict(complete=True,cycle_budgets=[16,32],shared_inference=True,checkpoint=plan['checkpoint'],stable_cycles=16,results={})
    for limit in [16,32]:
        folder=root/f'cycles_{limit}';folder.mkdir()
        shard=dict(indices=np.arange(2),correctness=np.arange(2)<correct,cycles=np.full(2,min(limit,17)))
        np.savez(folder/'rank_00000.npz',**shard)
        metrics=validate_complete_results([shard],2,limit,16)
        row=dict(complete=True,input_provenance=plan,checkpoint=plan['checkpoint'],stable_cycles=16,
            world_size=1,requested_max_cycles=limit,shared_inference_max_cycles=32,**metrics)
        (folder/'summary.json').write_text(json.dumps(row))
        combined['results'][str(limit)]=dict(summary=f'cycles_{limit}/summary.json',**metrics)
    (root/'summary.json').write_text(json.dumps(combined))


class CurveSummaryTests(unittest.TestCase):
    def fixture(self,root):
        for i in range(1,4):write_run(root/str(i),i,i-1)
        m=root/'curve_runs.json';m.write_text(json.dumps(dict(runs=[dict(family='HRM',training_seed=i,directory=str(i)) for i in range(1,4)])))
        return m

    def test_real_counts_and_sample_sd(self):
        with tempfile.TemporaryDirectory() as d:
            m=self.fixture(Path(d));result=summarize(m)
            for row in result['groups']['HRM']['curve']:
                self.assertEqual(row['accuracy_percent_mean'],50.)
                self.assertEqual(row['accuracy_percent_sample_sd'],50.)
            self.assertIn('not certified',result['limitations'])

    def test_summary_tamper_and_duplicate_seed_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=self.fixture(root)
            p=root/'1/cycles_32/summary.json';v=json.loads(p.read_text());v['correct']=1;p.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'actual shards'):summarize(m)
            v['correct']=0;p.write_text(json.dumps(v))
            v=json.loads(m.read_text());v['runs'][1]['training_seed']=1;m.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'Repeated training seed'):summarize(m)

    def test_missing_shard_and_impossible_stopping_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=self.fixture(root)
            (root/'2/cycles_16/rank_00000.npz').unlink()
            with self.assertRaises(TimeoutError):summarize(m)
        with self.assertRaisesRegex(ValueError,'enough predictions'):
            validate_complete_results([dict(indices=np.array([0]),correctness=np.array([True]),cycles=np.array([16]))],1,32,16)

    def test_duplicate_checkpoint_and_changed_cohort_rejected(self):
        for field,value,message in [('checkpoint_sha256','1'*64,'multiple training runs'),('dataset_order_sha256','different','same ordered cohort')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                root=Path(d);m=self.fixture(root);p=root/'2/run_config.json'
                plan=json.loads(p.read_text());plan[field]=value;p.write_text(json.dumps(plan))
                for limit in [16,32]:
                    p=root/f'2/cycles_{limit}/summary.json';row=json.loads(p.read_text());row['input_provenance']=plan;p.write_text(json.dumps(row))
                with self.assertRaisesRegex(ValueError,message):summarize(m)

    def test_selection_receipts_bound_into_curve_report(self):
        from hrm.training.records import record_evaluation,write_best_selection
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'Missing 16-cycle'):
                summarize(manifest,require_selection=True)
            specification=json.loads(manifest.read_text())
            for i,entry in enumerate(specification['runs'],1):
                training=root/f'train{i}';training.mkdir();ck=training/'best.pt';ck.write_bytes(f'weights{i}'.encode())
                log=training/'evaluation.jsonl'
                record_evaluation(log,seed=i,step=10,split='test',correct=i-1,total=2,cycles=16,weights='EMA')
                receipt=write_best_selection(ck,log,seed=i,step=10,split='test',cycles=16,weights='EMA')
                entry['selection_receipt']=f'train{i}/best.selection.json'
                p=root/f'{i}/run_config.json';plan=json.loads(p.read_text());plan['checkpoint_sha256']=receipt['checkpoint_sha256'];p.write_text(json.dumps(plan))
                for limit in [16,32]:
                    p=root/f'{i}/cycles_{limit}/summary.json';row=json.loads(p.read_text());row['input_provenance']=plan;p.write_text(json.dumps(row))
            manifest.write_text(json.dumps(specification))
            result=summarize(manifest,require_selection=True)
            self.assertTrue(all(r['scalar_selection_verified'] for r in result['groups']['HRM']['runs']))

if __name__=='__main__':unittest.main()
