import csv
import json
from pathlib import Path
import tempfile
import unittest
from hrm.paths import repository_root
from hrm.analyse.baselines.frontier_llm.prepare_records import prepare


class PreparationTest(unittest.TestCase):
    def test_reproducible_shared_sampling_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);paths={}
            for task,question in [('sudoku','0'*81),('maze','S G'+'#'*897)]:
                paths[task]=root/(task+'.csv')
                with paths[task].open('w',newline='') as stream:
                    w=csv.writer(stream);w.writerow(['question','answer'])
                    for _ in range(3):w.writerow([question,'unused'])
            profiles=repository_root()/'exp/sudoku/configs/frontier/profiles.json'
            one=prepare(root/'one',profiles,2,12345,paths)
            two=prepare(root/'two',profiles,2,12345,paths)
            self.assertEqual(one,two)
            self.assertFalse(one['api_called'])
            questions=[json.loads(l) for l in (root/'one/questions.jsonl').read_text().splitlines()]
            self.assertEqual(len(questions),16)
            for task in ('sudoku','maze'):
                grouped={model:[q['source_index'] for q in questions if q['model']==model and q['task']==task] for model in one['profiles']}
                self.assertEqual(len({tuple(v) for v in grouped.values()}),1)
            self.assertEqual(questions[0]['problem']['question'],'.'*81)
            with self.assertRaises(FileExistsError):prepare(root/'one',profiles,2,12345,paths)
            with self.assertRaises(ValueError):prepare(root/'too-many',profiles,4,12345,paths)
            self.assertFalse((root/'too-many').exists())
