import json
from pathlib import Path
import tempfile
import unittest
from hrm.analyse.baselines.frontier_llm.score_records import score


class FrontierRecordsTest(unittest.TestCase):
    def test_correct_wrong_missing_and_settings_separation(self):
        solution='534678912672195348198342567859761423426853791713924856961537284287419635345286179'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            questions=[dict(job_id=str(i),task='sudoku',model='fixture-model',settings={'max_tokens':100 if i<3 else 200},problem={'question':'.'*81}) for i in range(4)]
            answers=[dict(job_id=str(i),status='completed' if i!=2 else 'failed',final_content=json.dumps({'answer':solution}) if i in (0,3) else ('bad' if i==1 else '')) for i in range(4)]
            (root/'questions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in questions))
            (root/'final_answers.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in answers))
            result=score(root)
            self.assertEqual(result['n'],4)
            self.assertEqual([g['n'] for g in result['groups']],[3,1])
            self.assertEqual(result['groups'][0]['rates']['correct'],1/3)
            self.assertFalse(result['api_called'])
            (root/'final_answers.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in answers[:-1]))
            with self.assertRaisesRegex(ValueError,'IDs differ'):score(root)

    def test_shortest_maze_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);question='S G'+'#'*897;answer='SoG'+'#'*897
            q=dict(job_id='maze1',task='maze',model='fixture-model',settings={'max_tokens':100},problem={'question':question})
            a=dict(job_id='maze1',status='completed',final_content=json.dumps({'answer':answer}))
            (root/'questions.jsonl').write_text(json.dumps(q)+'\n')
            (root/'final_answers.jsonl').write_text(json.dumps(a)+'\n')
            self.assertEqual(score(root)['groups'][0]['counts']['correct'],1)
            (root/'questions.jsonl').write_text((json.dumps(q)+'\n')*2)
            with self.assertRaisesRegex(ValueError,'duplicate'):score(root)
