import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

from hrm_qwen.cot import evaluate as e

GOLD='534678912672195348198342567859761423426853791713924856961537284287419635345286179'


def args(root):
    return argparse.Namespace(records=str(root),dp_size=2,n=4,tp=1,offset=0,seed=12345,
        config=['test_hard:0:128:2'],data_dir=None,model_path=None,model=e.BASE_MODEL,
        revision=None,temperature=1.,top_p=.95,top_k=20,gpu_memory_utilization=.85,max_num_seqs=128)


class EvaluationTest(unittest.TestCase):
    def test_prepare_mock_execution_score_and_integrity(self):
        rows=[dict(index=i,prompt=[{'role':'user','content':'fixture'}],question='0'+GOLD[1:],answer=GOLD,empties=1) for i in range(2)]
        captured=[]
        class Tokenizer:
            @staticmethod
            def from_pretrained(*a,**kw):return Tokenizer()
            def apply_chat_template(self,*a,**kw):return 'prompt<think>\n'
            def __call__(self,*a,**kw):return {'input_ids':[1,2]}
        class LLM:
            def __init__(self,**kwargs):captured.append(kwargs)
            def generate(self,inputs,params,**kwargs):
                return [SimpleNamespace(outputs=[SimpleNamespace(text='reason</think>'+GOLD,token_ids=[1],finish_reason='stop')]+[SimpleNamespace(text='unfinished',token_ids=[1,2],finish_reason='length') for _ in range(params.n-1)]) for _ in inputs]
        tf=ModuleType('transformers');tf.AutoTokenizer=Tokenizer
        vl=ModuleType('vllm');vl.LLM=LLM;vl.SamplingParams=SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'base-eval'
            with patch.object(e,'eval_rows',return_value=rows):plan=e.prepare(args(root))
            with patch.dict(sys.modules,{'transformers':tf,'vllm':vl}):
                e.execute(root,0)
                with self.assertRaises(FileNotFoundError):e.score(root)
                e.execute(root,1)
                with self.assertRaises(FileExistsError):e.execute(root,0)
            m=e.score(root)['test_hard/empties0/budget128']
            self.assertEqual((m['n_puzzles'],m['n'],m['pass@1'],m['pass@4'],m['truncated_rate']),(2,8,.25,1.,.75))
            self.assertEqual([c['seed'] for c in captured],[12345,12346])
            self.assertTrue(all(c['revision']==e.BASE_REVISION for c in captured))
            from hrm.analyse.baselines.qwen.report import report
            student=Path(temp)/'sft-eval'
            student_args=args(student);student_args.config=['test_hard:0:65536:2']
            with patch.object(e,'eval_rows',return_value=rows):e.prepare(student_args)
            with patch.dict(sys.modules,{'transformers':tf,'vllm':vl}):
                e.execute(student,0);e.execute(student,1)
            summary=report(root,student,Path(temp)/'summary.json',base_budget=128,replicates=100,seed=12345)
            self.assertEqual(summary['counts']['outputs_per_condition'],8)
            self.assertEqual(summary['conditions']['instruct']['mean_percent'],25.)
            self.assertEqual(summary['conditions']['paired_sft_minus_instruct']['ci95_high_percent'],0.)
            self.assertFalse((student/'metrics.json').exists())  # report validates inputs without changing them
            from hrm.cli import main
            before=(Path(temp)/'summary.json').read_bytes()
            with patch.object(sys,'argv',['hrm','analyse','baselines.qwen.report','--experiment','E15','--records',temp,'--replicates','100','--seed','12345','--base-budget','128']):
                main()
            self.assertEqual((Path(temp)/'summary.json').read_bytes(),before)


            shard=root/'rank0.jsonl';original=shard.read_text();shard.write_text(original+original)
            done=root/'rank0.done.json';d=json.loads(done.read_text());d['result_sha256']=e.file_hash(shard);done.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'Duplicate'):e.score(root)
            (root/'questions.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError,'questions changed'):e.score(root)

    def test_invalid_checkpoint_and_sample_size(self):
        with tempfile.TemporaryDirectory() as temp:
            a=args(Path(temp)/'run');a.model_path=str(Path(temp)/'missing')
            with patch.object(e,'eval_rows',return_value=[{},{}]):
                with self.assertRaises(FileNotFoundError):e.prepare(a)
            a=args(Path(temp)/'run')
            with patch.object(e,'eval_rows',return_value=[]):
                with self.assertRaisesRegex(ValueError,'exceeds'):e.prepare(a)
