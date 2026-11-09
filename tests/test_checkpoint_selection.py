import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from hrm.training.records import record_evaluation,write_best_selection,verify_best_selection


class SelectionTests(unittest.TestCase):
    def test_receipt_binds_best_hash_and_survives_later_worse_or_tied_steps(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); ck=root/'best.pt'; log=root/'evaluation.jsonl'
            ck.write_bytes(b'actual saved checkpoint bytes')
            for step,correct in [(10,1),(20,2)]:
                record_evaluation(log,seed=1794189001,step=step,split='test',correct=correct,total=3,cycles=16,weights='EMA')
            receipt=write_best_selection(ck,log,seed=1794189001,step=20,split='test',cycles=16,weights='EMA')
            digest=hashlib.sha256(ck.read_bytes()).hexdigest()
            for step,correct in [(30,1),(40,2)]:
                record_evaluation(log,seed=1794189001,step=step,split='test',correct=correct,total=3,cycles=16,weights='EMA')
            selected=verify_best_selection(ck.with_suffix('.selection.json'),checkpoint_sha256=digest,seed=1794189001)
            self.assertEqual(selected['optimizer_step'],20)
            self.assertEqual(receipt['checkpoint_sha256'],digest)
            with self.assertRaisesRegex(ValueError,'hash differs'):
                verify_best_selection(ck.with_suffix('.selection.json'),checkpoint_sha256='different',seed=1794189001)
            record_evaluation(log,seed=1794189001,step=50,split='test',correct=3,total=3,cycles=16,weights='EMA')
            with self.assertRaisesRegex(ValueError,'Later/better'):
                verify_best_selection(ck.with_suffix('.selection.json'),checkpoint_sha256=digest,seed=1794189001)

    def test_reject_wrong_budget_weight_and_modified_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ck=root/'best.pt';log=root/'evaluation.jsonl';ck.write_bytes(b'checkpoint')
            record_evaluation(log,seed=1794189002,step=10,split='test',correct=2,total=3,cycles=32,weights='raw')
            receipt=write_best_selection(ck,log,seed=1794189002,step=10,split='test',cycles=32,weights='raw')
            path=ck.with_suffix('.selection.json')
            with self.assertRaisesRegex(ValueError,'protocol'):
                verify_best_selection(path,checkpoint_sha256=receipt['checkpoint_sha256'],seed=1794189002)
            log.write_text(log.read_text().replace('"correct": 2','"correct": 1'))
            with self.assertRaisesRegex(ValueError,'records changed'):
                verify_best_selection(path,checkpoint_sha256=receipt['checkpoint_sha256'],seed=1794189002,cycles=32,weights='raw')

if __name__=='__main__':unittest.main()
