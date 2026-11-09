import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hrm.analyse.dynamics.h_update.run import main, boundary, branch
from hrm.analyse.dynamics.jacobian.models import HRM, analysis_architecture


class HUpdateTests(unittest.TestCase):
    def test_same_start_and_independent_recurrence(self):
        config=dict(arch=dict(name='hrm.models.hrm@HRM',seq_len=82,vocab_size=10,hidden_size=8,
             intermediate_size=16,head_dim=4,num_layers=1,norm_eps=1e-6,rope_theta=10000.,
             forward_dtype='bfloat16',H_cycles=2,L_cycles=6,bptt=True))
        torch.manual_seed(23)
        model=HRM(analysis_architecture(config)).eval()
        x=torch.zeros(2,82,dtype=torch.long);x[1,1]=1;y=torch.ones_like(x)
        state=boundary(model,x,phase=2,l_steps=4,tail_steps=3)
        initial=state['l0'].clone()
        with torch.inference_mode():
            for alpha in (0,1):
                arrays,summary=branch(model,x,y,state,1,alpha,l_steps=4,follow_h=2)
                h=(state['old'].float()+alpha*(state['new'].float()-state['old'].float())).bfloat16()
                l=initial.clone();emb=model.embed(x);boards=[];nearest=[]
                for i in range(5):
                    boards.append(model.lm_head(model.H_level(h+l)).argmax(-1)[1,1:].numpy())
                    nearest.append(float((state['tail'][:,1].float()-l[1].float()).square().mean((1,2)).min().sqrt()))
                    if i<4:l=model.L_level(l+h+emb)
                h=model.H_level(h+l)
                follow=[int((model.lm_head(h).argmax(-1)[1,1:]!=y[1,1:]).sum())]
                for _ in range(2):
                    for _ in range(6):l=model.L_level(l+h+emb)
                    h=model.H_level(h+l)
                    follow.append(int((model.lm_head(h).argmax(-1)[1,1:]!=y[1,1:]).sum()))
                np.testing.assert_array_equal(arrays['boards'],boards)
                np.testing.assert_allclose(arrays['nearest'],nearest,atol=1e-7)
                np.testing.assert_array_equal(arrays['states'][0],initial[1].float().numpy())
                self.assertEqual(summary['follow_errors'],follow)
                self.assertTrue(torch.equal(initial,state['l0']))
                if alpha:self.assertTrue(summary['independent_next_boundary_bitwise_equal'])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cp=root/'model.pt';data=root/'test.csv';out=root/'records'
            torch.save(model.state_dict(),cp);(root/'model_config.json').write_text(json.dumps(config))
            data.write_text('question,answer\n'+'.'*81+','+'123456789'*9+'\n')
            args=['--checkpoint',str(cp),'--dataset',str(data),'--output-dir',str(out),'--seed','1794196000',
                  '--boards','1','--phase','2','--l-steps','4','--tail-steps','3','--follow-h','2','--device','cpu']
            main(args+['--prepare-only']);main(args)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertTrue(manifest['same_L_start'])
            self.assertEqual(manifest['seed'],1794196000)
            with self.assertRaises(FileExistsError):main(args)
            with self.assertRaisesRegex(ValueError,'Changed'):main(args+['--seed','1794196001','--prepare-only'])
            state['next_L']=state['next_L']+1
            with self.assertRaisesRegex(RuntimeError,'independent next'):branch(model,x,y,state,1,1,4,2)


if __name__=='__main__':unittest.main()
