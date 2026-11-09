import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from hrm.models.hrm import HRM
from hrm.analyse.dynamics.constraint_readout.run import hard, checks, excess_error, main, collect
from hrm.analyse.dynamics.constraint_readout.summarize import main as summarize


class ConstraintReadoutTests(unittest.TestCase):
    def test_constraints_against_independent_enumeration(self):
        checks();rng=np.random.default_rng(12);boards=rng.integers(1,10,(16,81));clues=rng.integers(0,10,(16,81))
        expected=[]
        for board,clue in zip(boards,clues):
            grid=board.reshape(9,9);bad=[]
            for r in range(9):
                for c in range(9):
                    digit=grid[r,c];box=grid[r//3*3:r//3*3+3,c//3*3:c//3*3+3]
                    bad.append((grid[r]==digit).sum()>1 or (grid[:,c]==digit).sum()>1 or (box==digit).sum()>1)
            cb=(clue>0)&(board!=clue);expected.append([sum(bad),sum(np.asarray(bad)|cb),sum(cb)])
        np.testing.assert_array_equal(torch.stack(hard(torch.tensor(boards),torch.tensor(clues)),-1),expected)
        p=torch.zeros(1,81,9);p[:,:,0]=1
        self.assertEqual(excess_error(p).item(),216)

    def test_fresh_collection_fit_and_summary(self):
        arch=dict(name='hrm.models.hrm@HRM',seq_len=82,vocab_size=10,hidden_size=8,intermediate_size=16,head_dim=4,
                  num_layers=1,norm_eps=1e-6,rope_theta=10000.,forward_dtype='bfloat16',H_cycles=2,L_cycles=6,bptt=True)
        torch.manual_seed(17);model=HRM(arch).eval().requires_grad_(False)
        x=torch.zeros(2,82,dtype=torch.long);x[1,2]=3
        data,_=collect(model,[(x,x,torch.arange(2))],1,'cpu')
        with torch.inference_mode():
            e=model.embed(x);h=model.zH_init.expand_as(e).clone();l=model.zL_init.expand_as(e).clone()
            for i in range(12):
                torch.testing.assert_close(data['features'][:,i],torch.cat([l.mean(1),h.mean(1)],-1),rtol=0,atol=0)
                nxt=model.L_level(l+h+e)
                torch.testing.assert_close(data['delta'][:,i],(nxt-l).float().flatten(1).norm(dim=-1),rtol=0,atol=0)
                l=nxt
                if (i+1)%6==0:h=model.H_level(h+l)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);torch.save(model.state_dict(),root/'model.pt')
            (root/'model_config.json').write_text(json.dumps(dict(arch=arch)))
            answer=''.join(str((r*3+r//3+c)%9+1) for r in range(9) for c in range(9))
            rows=['question,answer']
            for i in range(8):rows.append(''.join(answer[j] if j<=i else '0' for j in range(81))+','+answer)
            (root/'test.csv').write_text('\n'.join(rows)+'\n')
            args=['--checkpoint',str(root/'model.pt'),'--dataset',str(root/'test.csv'),'--device','cpu','--boards','8','--cycles','1',
                  '--bootstrap-puzzles','8','--permutation-puzzles','9','--output-dir',str(root/'out')]
            main(args+['--prepare-only']);plan=(root/'out/run_config.json').read_text();main(args)
            self.assertEqual(plan,(root/'out/run_config.json').read_text())
            result=json.loads((root/'out/results.json').read_text());self.assertEqual(result['split'],[4,2,2])
            self.assertEqual(result['observations_per_puzzle'],12)
            summarize(['--records',str(root/'out')]);self.assertTrue((root/'out/metrics.csv').is_file())
            self.assertFalse(list((root/'out').glob('*.png')))
