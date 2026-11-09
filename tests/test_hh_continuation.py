import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import pandas as pd
import torch
from hrm.models.ablations import HRM
from hrm.training.optimizer import AdamATan2
from hrm.training.steps import train_step
from hrm.training.hh_continuation import main, restore
from hrm.experiments import recipe_config, recipes


class ContinuationTests(unittest.TestCase):
    def test_restore_and_matched_branches(self):
        arch=dict(seq_len=82,vocab_size=10,hidden_size=8,intermediate_size=16,head_dim=4,
            num_layers=1,norm_eps=1e-6,rope_theta=10000.,forward_dtype='bfloat16',H_cycles=2,L_cycles=6,bptt=True,hh_link=True)
        cfg=dict(arch=arch,cycles_per_data=16,data=dict(name='hrm.tasks.sudoku_stochastic_h',augment=False,repeat=1))
        torch.manual_seed(1794199000);model=HRM(arch)
        opt=AdamATan2(model.parameters(),lr=torch.tensor(1e-4),betas=(.9,.95),weight_decay=1.,ema=.999)
        x=torch.zeros(2,82,dtype=torch.long);y=torch.ones_like(x);y[:,0]=0
        train_step(model,model.initial_carry,opt,x,y)
        saved=copy.deepcopy(dict(model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),epoch=19,step=1,world_size=8))
        restored=HRM(arch);other=AdamATan2(restored.parameters(),ema=.999)
        restore(restored,other,copy.deepcopy(saved))
        for (p,st),(q,tt) in zip(opt.state.items(),other.state.items()):
            self.assertTrue(torch.equal(p,q))
            for key in ('step','exp_avg','exp_avg_sq','param_ema'):self.assertTrue(torch.equal(st[key],tt[key]))
        train_step(model,model.initial_carry,opt,x,y);train_step(restored,restored.initial_carry,other,x,y)
        for p,q in zip(model.parameters(),restored.parameters()):self.assertTrue(torch.equal(p,q))
        with self.assertRaisesRegex(ValueError,'full training checkpoint'):restore(restored,other,saved['model_state_dict'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';data.mkdir();cp=root/'latest_train.pt'
            torch.save(saved,cp);(root/'model_config.json').write_text(json.dumps(cfg))
            answer=''.join(str((r*3+r//3+c)%9+1) for r in range(9) for c in range(9))
            rows=[dict(question=str(i)+'.'*80,answer=answer) for i in range(1,5)]
            pd.DataFrame(rows[:2]).to_csv(data/'train.csv',index=False);pd.DataFrame(rows[2:]).to_csv(data/'test_hard.csv',index=False)
            plans=[];initial=[];last=[]
            for branch in ('full','deleted'):
                out=root/branch;args=['--source',str(cp),'--dataset',str(data),'--output-dir',str(out),
                    '--branch',branch,'--seed','1794199000','--epochs','1','--global-batch','2','--evaluation-puzzles','2','--device','cpu']
                main(args+['--prepare-only']);main(args)
                plans.append(json.loads((out/'run_config.json').read_text()))
                manifest=json.loads((out/'manifest.json').read_text());self.assertEqual(manifest['status'],'complete');self.assertEqual(manifest['steps'],16)
                metrics=[json.loads(l) for l in (out/'evaluation.jsonl').read_text().splitlines()]
                self.assertEqual([r['epoch'] for r in metrics],[0,1]);self.assertEqual([r['total'] for r in metrics],[2,2])
                from hrm.analyse.training_summary import summarize
                self.assertEqual(summarize(out,'test_hard',16)['runs'][0]['evaluations'],2)
                initial.append(torch.load(out/'epoch_0.pt',weights_only=True));last.append(torch.load(out/'latest_train.pt',weights_only=True))
                with self.assertRaises(FileExistsError):main(args)
            distributed=root/'distributed'
            subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
                '-m','hrm.training.hh_continuation','--source',str(cp),'--dataset',str(data),
                '--output-dir',str(distributed),'--branch','full','--seed','1794199000',
                '--epochs','1','--global-batch','2','--evaluation-puzzles','2','--device','cpu'],
                check=True,capture_output=True,text=True,timeout=60)
            self.assertEqual(json.loads((distributed/'manifest.json').read_text())['status'],'complete')
            distributed_rows=[json.loads(l) for l in (distributed/'evaluation.jsonl').read_text().splitlines()]
            self.assertEqual([r['total'] for r in distributed_rows],[2,2])
            self.assertEqual(plans[0]['evaluation_ids'],plans[1]['evaluation_ids'])
            for k in initial[0]:self.assertTrue(torch.equal(initial[0][k],initial[1][k]))
            self.assertTrue(any(not torch.equal(last[0]['model_state_dict'][k],last[1]['model_state_dict'][k]) for k in initial[0]))
            for record in last:
                self.assertTrue(all(int(s['step'])==17 for s in record['optimizer_state_dict']['state'].values()))

    def test_continuation_recipes_cannot_silently_start_fresh(self):
        for name,recipe in recipes('E30').items():
            if name.startswith('seed_'):
                for fresh in (False,True):
                    with self.assertRaisesRegex(ValueError,'Matched continuation'):recipe_config(recipe,fresh)
