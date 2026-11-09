import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hrm.paths import repository_root
from hrm.analyse.baselines.frontier_llm.prepare_records import prepare as prepare_inputs
from hrm.analyse.baselines.frontier_llm.execute_records import prepare, main
from hrm.analyse.baselines.frontier_llm.score_records import score
from hrm.analyse.baselines.frontier_llm import frozen_code,run


class ExecutionTest(unittest.TestCase):
    def test_execute_and_resume_use_first_terminal_without_new_call(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);paths={}
            for task,q in [('sudoku','.'*81),('maze','S G'+'#'*897)]:
                paths[task]=root/(task+'.csv')
                with paths[task].open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['question','answer']);w.writerow([q,'unused'])
            profiles=json.loads((repository_root()/'exp/sudoku/configs/frontier/profiles.json').read_text())
            model=next(iter(profiles));p=root/'profiles.json';p.write_text(json.dumps({model:profiles[model]}))
            prepare_inputs(root/'inputs',p,1,12345,paths)
            jobs,manifest=prepare(root/'inputs')
            def response(job,client,key,timeout):
                return {**{k:job[k] for k in ('task','model','case_id','problem_sha256','payload_sha256')},
                        'status':'completed','http_status':200,'finish_reasons':['stop'],'native_finish_reasons':[],
                        'error_codes':[],'configuration_check':{'status':'verified','issues':[]},
                        'final_content':'{"answer":"wrong"}','correct':False,'strict_output_format':False,
                        'usage':{},'elapsed_seconds':.01}
            original=frozen_code.load_original()
            with patch.dict(os.environ,{'OPENROUTER_API_KEY':'mock-not-a-real-key'}),patch.object(original.transport,'run_one',side_effect=response) as fake:
                result=run.execute(jobs,manifest,root/'execution',workers=1,max_attempts=2)
                self.assertEqual(result['finalized'],2)
                self.assertEqual(fake.call_count,2) # Wrong answers must not trigger retries.
                again=run.execute(jobs,manifest,root/'execution',workers=1,max_attempts=2,resume=True)
                self.assertEqual(again['unresolved'],0)
                self.assertEqual(fake.call_count,2)
                with patch('sys.argv',['execute_records','--records',str(root/'inputs'),'--output',str(root/'execution'),'--execute','--resume','--workers','1','--max-attempts','2']):
                    main()
                self.assertEqual(score(root/'execution')['n'],2)
                self.assertEqual(fake.call_count,2)
                with self.assertRaises(ValueError):run.execute(jobs,manifest,root/'execution',workers=1,max_attempts=3,resume=True)
            saved=json.loads((root/'execution/execution_plan.json').read_text())
            self.assertEqual(saved['retry_policy']['max_attempts'],2)
            q=root/'inputs/questions.jsonl';q.write_text(q.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'Prepared input changed'):prepare(root/'inputs')

    def test_provider_drift_blocks_terminal_selection(self):
        original=frozen_code.load_original()
        profiles=json.loads((repository_root()/'exp/sudoku/configs/frontier/profiles.json').read_text())
        for model,profile in profiles.items():
            params={}
            for requirement in profile['upstream_requirements']:
                parts=requirement['paths'][0].split('.')
                target=params
                for part in parts[:-1]:target=target.setdefault(part,{})
                target[parts[-1]]=requirement['accepted_values'][0]
            result=dict(status='completed',http_status=200,finish_reasons=['stop'],native_finish_reasons=[],error_codes=[],
                        observed_models=[profile['expected_response_models'][0]],observed_providers=profile['expected_provider_names'][:1],
                        upstream_debug=[dict(provider=profile['expected_provider_names'][0],upstream_parameters=params)])
            job=dict(model_profile=profile,call_settings=profile['call_settings'])
            result['configuration_check']=original.transport.check_configuration(job,result)
            self.assertEqual(result['configuration_check']['status'],'verified',model)
            result['observed_providers']=['unexpected-provider']
            result['configuration_check']=original.transport.check_configuration(job,result)
            self.assertEqual(original.retry.classify_attempt(result)['kind'],'blocked_engineering')
