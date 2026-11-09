import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from hrm_qwen.cot import generate as g
from hrm_qwen.cot.generation_config import trace_id
from hrm_qwen.cot.train import load_examples

GOLD='534678912672195348198342567859761423426853791713924856961537284287419635345286179'
QUESTION='0'*5+GOLD[5:]


class GenerationTest(unittest.TestCase):
    def invoke(self,out,*extra):
        argv=['generate','--out',str(out),'--count','1','--seed','12345','--workers','1',*extra]
        with patch.object(sys,'argv',argv),patch.object(g,'load_puzzles',return_value=[(QUESTION,GOLD)]):
            g.main()

    def test_prepare_generate_resume_and_hinted_identity(self):
        response={'choices':[{'message':{'reasoning_content':'reason','content':GOLD},'finish_reason':'stop'}],
                  'usage':{'completion_tokens':90,'cost':.01},'model':'reported-model'}
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'train.jsonl'
            with patch.object(g,'api_key',side_effect=AssertionError('preparation read credentials')):
                self.invoke(out)
            self.assertFalse(out.exists())
            with patch.object(g,'api_key',return_value='mock'),patch.object(g,'ask',return_value=response) as ask:
                self.invoke(out,'--execute')
                self.invoke(out,'--execute')
                self.assertEqual(ask.call_count,1)
            original=json.loads(out.read_text())
            self.assertTrue(original['exact']);self.assertEqual(original['source_index'],0)
            hinted=Path(temp)/'hinted.jsonl'
            with patch.object(g,'api_key',return_value='mock'),patch.object(g,'ask',return_value=response):
                self.invoke(hinted,'--empties','2','--execute')
            variant=json.loads(hinted.read_text())
            self.assertEqual(variant['index'],original['index'])
            self.assertNotEqual(variant['variant_id'],original['variant_id'])
            self.assertEqual(variant['question'].count('0'),2)
            self.assertEqual(original['variant_id'],trace_id(QUESTION,GOLD))
            class Tokenizer:
                def apply_chat_template(self,messages,**kwargs):return messages[0]['content']
                def __call__(self,text,**kwargs):return {'input_ids':list(text.encode())}
                def convert_tokens_to_ids(self,token):return 999
            combined,stats=load_examples(Tokenizer(),[str(out),str(hinted)],131072,1)
            base,_=load_examples(Tokenizer(),[str(out)],131072,1)
            self.assertEqual(combined,base)
            self.assertEqual(stats['n_records'],1)
            with self.assertRaisesRegex(ValueError,'changed'):self.invoke(out,'--empties','2')

    def test_rate_limits_bounded_and_request_shape(self):
        limited=type('Response',(),{'status_code':429,'text':'rate limited'})()
        with patch.object(g.requests,'post',return_value=limited) as post,patch.object(g.time,'sleep'):
            result=g.ask('deepseek','mock','teacher',[{'role':'user','content':'puzzle'}],100,90,1.,10.,None,retries=3)
        self.assertEqual(post.call_count,3)
        self.assertIn('error',result)
        body=post.call_args.kwargs['json']
        self.assertEqual(body['thinking'],{'type':'enabled'})
        self.assertEqual(body['max_tokens'],100)

    def test_selected_cohort_does_not_expand_on_resume(self):
        response={'choices':[{'message':{'reasoning_content':'reason','content':GOLD},'finish_reason':'stop'}],'usage':{}}
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'selected.jsonl'
            argv=['generate','--out',str(out),'--count','2','--select-count','1','--seed','12345','--workers','1','--execute']
            rows=[(QUESTION,GOLD),('0'*6+GOLD[6:],GOLD)]
            with patch.object(sys,'argv',argv),patch.object(g,'load_puzzles',return_value=rows),patch.object(g,'api_key',return_value='mock'),patch.object(g,'ask',return_value=response) as ask:
                g.main();g.main()
            self.assertEqual(ask.call_count,1)
            self.assertEqual(len(out.read_text().splitlines()),1)
            self.assertEqual(len(json.loads(Path(str(out)+'.config.json').read_text())['selected_indices']),1)
