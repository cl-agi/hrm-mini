import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from hrm.analyse.sudoku.single_clue.evaluate import load_pairs,main as evaluate_pairs
from hrm.analyse.dynamics.jacobian.models import HRM,analysis_architecture


class FreshEntries(unittest.TestCase):
    def test_public_single_clue_parent_join(self):
        parent=dict(id='train:3',puzzle='1'+'0'*80,solution='1'*81)
        edited=dict(id='edited',parent_id='train:3',puzzle='2'+'0'*80,solution='2'*81)
        def data(task,split):return [edited] if split=='test_single_clue' else [parent] if split=='train' else []
        with patch('hrm.hub.dataset',side_effect=data):
            result=load_pairs();self.assertEqual(result['cell'].tolist(),[0]);self.assertEqual(result['parent_ids'].tolist(),['train:3'])

    def test_local_single_clue_and_answer_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir()
            config={'arch':dict(name='hrm@HRM',hidden_size=8,intermediate_size=16,head_dim=4,num_layers=1,norm_eps=1e-6,rope_theta=10000.,forward_dtype='float32',H_cycles=2,L_cycles=6,bptt=True)}
            model=HRM(analysis_architecture(config));cp=root/'model.pt';torch.save(model.state_dict(),cp)
            (root/'model_config.json').write_text(json.dumps(config))
            base=np.array([(r*3+r//3+c)%9+1 for r in range(9) for c in range(9)])
            import csv
            with (data/'test.csv').open('w') as f:
                w=csv.writer(f);w.writerow(['question','answer'])
                for i in range(12):
                    board=(base+i%9-1)%9+1;q=board.copy();q[i:i+5]=0;w.writerow([''.join(map(str,q)),''.join(map(str,board))])
            old=np.stack([base]*2);new=old.copy();new[:,0]=0
            pairs=root/'pairs.npz';np.savez(pairs,question_old=old,question_new=new,solution_old=old,solution_new=old)
            evaluate_pairs(['--checkpoint',str(cp),'--dataset-path',str(pairs),'--output',str(root/'pair-out'),'--device','cpu','--cycles','1'])
            self.assertEqual(json.loads((root/'pair-out/summary.json').read_text())['results']['num_pairs'],2)
            from hrm.analyse.sudoku.single_clue.prepare_statistics import main as statistics
            statistics(['--records',str(root/'pair-out')])
            from hrm.analyse.sudoku.single_clue_controls.verify_pair_metrics import verify
            self.assertTrue(verify(root/'pair-out')['passed'])
            from hrm.analyse.dynamics.answer_error.collect_fit import main
            main(['--checkpoint',str(cp),'--dataset',str(data),'--boards','8','--cycles','1','--epochs','1','--bootstrap','2','--output',str(root/'ce'),'--device','cpu'])
            self.assertTrue((root/'ce/probes/ce/results.json').exists())
            self.assertTrue((root/'ce/probes/mse/results.json').exists())

    def test_easy_preparation_recipes_and_reuse(self):
        from hrm.analyse.sudoku.difficulty.prepare import main
        with tempfile.TemporaryDirectory() as tmp:
            for tiny in (False,True):
                for model in ('hrm','rt'):
                    out=Path(tmp)/f'{model}-{tiny}'
                    def prepare(cmd,check):
                        self.assertIn('--fresh-seed',cmd)
                        self.assertIn('--prepare-only',cmd)
                        recipe=cmd[cmd.index('--recipe')+1]
                        self.assertEqual('resampled' in recipe,not tiny)
                        out.mkdir();(out/'resolved_config.json').write_text('{}')
                    args=['--model',model,'--output',str(out)]
                    with patch('hrm.analyse.sudoku.difficulty.prepare.subprocess.run',side_effect=prepare) as runner:
                        main(args,tiny=tiny);main(args,tiny=tiny)
                        self.assertEqual(runner.call_count,1)
                        with self.assertRaises(ValueError):main(args+['--data-seed','123'],tiny=tiny)

    def test_help_without_records(self):
        import subprocess,sys,os
        with tempfile.TemporaryDirectory() as tmp:
            env=dict(os.environ,HRM_WORKDIR=tmp);env.pop('HRM_ANALYSIS_RECORDS',None)
            for mod in ('sudoku.difficulty.prepare','sudoku.difficulty_1k.prepare','sat.graph_coloring.recurrent','sat.constraint_intervention.prepare','dynamics.constraint_intervention.intervention','dynamics.answer_error.collect_fit'):
                p=subprocess.run([sys.executable,'-m','hrm.analyse.'+mod,'--help'],env=env,capture_output=True,text=True)
                self.assertEqual(p.returncode,0,(mod,p.stderr))

if __name__=='__main__':unittest.main()
