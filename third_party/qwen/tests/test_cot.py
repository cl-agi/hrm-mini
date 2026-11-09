import argparse
import json
from pathlib import Path
import tempfile
import unittest
from hrm_qwen.cot.train import build_example, load_examples, collate
from hrm_qwen.cot.run_config import freeze_run


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content'] + '<think>\n'

    def __call__(self, text, **kwargs):
        return {'input_ids': list(text.encode())}

    def convert_tokens_to_ids(self, token):
        assert token == '<|im_end|>'
        return 999


class TeacherCoTTest(unittest.TestCase):
    def test_full_completion_labels_and_filtering(self):
        tok = Tokenizer()
        rec = dict(index=0, question='0'*81, reasoning='reason', content='answer', exact=True)
        prompt, completion = build_example(tok, rec)
        self.assertEqual(bytes(completion[:-1]).decode(), 'reason\n</think>\n\nanswer')
        batch = collate([dict(input_ids=prompt+completion, n_prompt=len(prompt))])
        self.assertTrue((batch['labels'][0,:len(prompt)] == -100).all())
        self.assertEqual(batch['labels'][0,len(prompt):].tolist(), completion)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'traces.jsonl'
            rows = [dict(rec, exact=False), rec, dict(rec, index=1, reasoning='x'*5000), dict(rec,index=2,exact=False)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            examples, stats = load_examples(tok,[str(path)],1000,2)
        self.assertEqual(len(examples),2)
        self.assertEqual((stats['n_records'],stats['n_used'],stats['dropped_wrong'],stats['dropped_long']),(3,1,1,1))

    def test_frozen_seed_input_and_launch_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp)/'trace.jsonl'; trace.write_text('{}\n')
            args = argparse.Namespace(output_dir=str(Path(temp)/'run'),traces=[str(trace)],seed=None,prepare_only=True,epochs=2,grad_accum=4,max_len=131072)
            first = freeze_run(args)
            args.seed = None
            self.assertEqual(freeze_run(args),first)
            args.prepare_only=False
            freeze_run(args)
            with self.assertRaises(FileExistsError): freeze_run(args)
            args.prepare_only=True
            trace.write_text('{"changed":true}\n')
            with self.assertRaises(ValueError): freeze_run(args)
