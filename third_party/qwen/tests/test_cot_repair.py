import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from hrm_qwen.cot import repair as r
from hrm_qwen.cot.train import load_examples

GOLD='534678912672195348198342567859761423426853791713924856961537284287419635345286179'


class RepairTest(unittest.TestCase):
    def test_cut_guards(self):
        self.assertIsNone(r.first_wrong_placement('Suppose r1c1=1.',GOLD,None))
        self.assertIsNone(r.first_wrong_placement('r1c1=1. Later r1c1=5.',GOLD,None))
        self.assertIsNone(r.first_wrong_placement('r0c0=9. r9c0=1.',GOLD,None))
        text='Verified prefix.\nr1c1=1.'
        self.assertEqual(r.cut_point(text,r.first_wrong_placement(text,GOLD,None)),len('Verified prefix.\n'))
        # With an observed wrong final grid, its committed wrong cell wins over retraction heuristic.
        pred='1'+GOLD[1:]
        self.assertEqual(r.first_wrong_placement('r1c1=1. Later r1c1=5.',GOLD,pred),0)

    def test_prepare_repair_resume_and_train_selection(self):
        class Tok:
            def apply_chat_template(self,messages,**kwargs):return messages[0]['content']
            def __call__(self,text,**kwargs):return {'input_ids':list(text.encode())}
            def convert_tokens_to_ids(self,token):return 999
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source.jsonl';out=Path(temp)/'repair.jsonl'
            source.write_text(json.dumps(dict(index=0,question='0'+GOLD[1:],answer=GOLD,reasoning='Verified prefix.\nr1c1=1.',content='1'+GOLD[1:],exact=False,parsed=True,seed=12345))+'\n')
            argv=['repair','--source',str(source),'--out',str(out),'--seed','12345','--workers','1']
            with patch.object(sys,'argv',argv),patch.object(r,'api_key',side_effect=AssertionError('prepare read credentials')):r.main()
            self.assertFalse(out.exists())
            response={'choices':[{'message':{'content':GOLD},'finish_reason':'stop'}],'usage':{}}
            with patch.object(sys,'argv',argv+['--execute']),patch.object(r,'api_key',return_value='mock'),patch.object(r,'continue_from',return_value=response) as call:
                r.main();r.main();self.assertEqual(call.call_count,1)
                self.assertEqual(call.call_args.args[3],'Verified prefix.\n')
            record=json.loads(out.read_text());self.assertTrue(record['exact']);self.assertTrue(record['repaired'])
            examples,stats=load_examples(Tok(),[str(source),str(out)],131072,1)
            self.assertEqual(len(examples),1);self.assertEqual(stats['n_repaired'],1)
            source.write_text(source.read_text()+'\n')
            with patch.object(sys,'argv',argv),self.assertRaises((ValueError,json.JSONDecodeError)):r.main()

    def test_bounded_rate_limit(self):
        response=type('Response',(),{'status_code':429,'text':'rate limited'})()
        with patch.object(r.requests,'post',return_value=response) as post,patch.object(r.time,'sleep'):
            result=r.continue_from('mock','teacher','prompt','prefix',4096,1.,10.)
        self.assertEqual(post.call_count,4);self.assertIn('error',result)
        self.assertTrue(post.call_args.kwargs['json']['messages'][1]['prefix'])
