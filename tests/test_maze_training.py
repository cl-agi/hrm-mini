import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hrm.paths import repository_root
from hrm.training import maze


class MazePreparationTests(unittest.TestCase):
    def test_prepare_freezes_seed_and_rejects_changed_family_or_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(maze,'input_identity',return_value={'train':'a'}),patch('time.time',return_value=1794178000):
                first=maze.prepare(directory,'HRM')
                self.assertEqual(first['seed'],1794178000)
                self.assertEqual(maze.prepare(directory,'HRM'),first)
                with self.assertRaisesRegex(ValueError,'differs'):maze.prepare(directory,'RT')
                self.assertEqual(maze.read_config(directory),first)
            with patch.object(maze,'input_identity',return_value={'train':'changed'}):
                with self.assertRaisesRegex(ValueError,'Data changed'):maze.read_config(directory)


class MazeUpdateTests(unittest.TestCase):
    def test_shared_update_matches_historical_full_token_equation(self):
        import torch
        import torch.nn.functional as F
        from hrm.training.steps import train_step
        from hrm.training.optimizer import AdamATan2
        # Independent model fixtures must not inherit Dynamo's variant cache
        # from earlier tests in a full-suite run. Keep compilation enabled.
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        source=repository_root()/'hrm/analyse/maze/extrapolation/train_abcd.py'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train_step')
        scope={'torch':torch,'F':F}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        torch.set_num_threads(1)
        for family in ('HRM','RT'):
            config=json.loads((repository_root()/f'exp/maze/configs/extrapolation/{family.lower()}.json').read_text())
            config['architecture'].update(hidden_size=32,intermediate_size=64,head_dim=8,num_layers=1,
                                          allowed_sizes=[2],seq_len=5,forward_dtype='float32')
            a=maze.make_model(config);b=copy.deepcopy(a)
            oa=AdamATan2(a.parameters(),lr=1e-4,weight_decay=.5,ema=.999)
            ob=AdamATan2(b.parameters(),lr=1e-4,weight_decay=.5,ema=.999)
            x=torch.tensor([[0,2,3,4,1]]);y=torch.tensor([[0,5,3,4,1]])
            ca,cb=a.initial_carry,b.initial_carry
            for _ in range(2):
                ca,ma=train_step(a,ca,oa,x,y)
                cb,mb=scope['train_step'](b,cb,ob,x,y)
                self.assertEqual(ma['loss'].item(),mb['loss'].item())
            for key,value in a.state_dict().items():
                self.assertTrue(torch.equal(value,b.state_dict()[key]),(family,key))
            oa.swap_ema();ob.swap_ema()
            for key,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[key]))
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'last.pt';maze.save_weights(a,path)
                restored=maze.make_model(config)
                restored.load_state_dict(torch.load(path,weights_only=True))
                rows=[dict(id='tiny',maze=['SG','##'],solution=['SG','##'])]
                config['dataset_path']=None
                config['evaluation'].update(sizes=[2],examples_per_dataset=1,include_control=False,inference_cycles=1)
                with patch.object(maze,'load_rows',return_value=rows):
                    result=maze.evaluate(restored,config,device='cpu',batch_size=1)
                self.assertEqual(result['test_2']['total'],1)
                self.assertEqual(result['test_2']['metric'],'shortest_path_accuracy')


if __name__=='__main__':unittest.main()
