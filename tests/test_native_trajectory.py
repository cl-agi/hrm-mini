import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from hrm.models.hrm import HRM
from hrm.analyse.dynamics.perturbation_recovery.native import trajectory,main


class NativeTrajectoryTests(unittest.TestCase):
    def test_readonly_lookahead_and_cli(self):
        arch=dict(name='hrm.models.hrm@HRM',seq_len=82,vocab_size=10,hidden_size=8,intermediate_size=16,head_dim=4,num_layers=1,norm_eps=1e-6,rope_theta=10000.,forward_dtype='bfloat16',H_cycles=2,L_cycles=6,bptt=True)
        torch.manual_seed(1794200000);model=HRM(arch).eval().requires_grad_(False)
        x=torch.zeros(2,82,dtype=torch.long);x[1,3]=4;y=torch.ones_like(x)
        arrays,report=trajectory(model,x,y,4)
        self.assertEqual(arrays['H'].shape,(29,2,82,8));self.assertEqual(report['native_cycle_bitwise_checks'],2)
        with torch.inference_mode():
            emb=model.embed(x);h=model.zH_init.expand_as(emb).clone();l=model.zL_init.expand_as(emb).clone()
            look=[];committed=[]
            for phase in range(4):
                for step in range(6):
                    l=model.L_level(l+h+emb)
                    look.append(model.lm_head(model.H_level(h+l)).argmax(-1)[:,1:].numpy())
                    np.testing.assert_array_equal(arrays['H'][phase*7+step+1],h.float().numpy())
                    np.testing.assert_array_equal(arrays['L'][phase*7+step+1],l.float().numpy())
                h=model.H_level(h+l);committed.append(model.lm_head(h).argmax(-1)[:,1:].numpy())
                np.testing.assert_array_equal(arrays['H'][(phase+1)*7],h.float().numpy())
            np.testing.assert_array_equal(arrays['lookahead'],look);np.testing.assert_array_equal(arrays['committed'],committed)
            # A synthetic target equal to the first lookahead checks timing conversion.
            target=torch.cat((torch.zeros(2,1,dtype=torch.long),torch.tensor(arrays['lookahead'][0])),1)
            changed,report2=trajectory(model,x,target,4)
            for key in ('H','L','lookahead','committed'):np.testing.assert_array_equal(changed[key],arrays[key])
            self.assertTrue(all(r['first_correct_lookahead_L']==1 and r['first_correct_lookahead_phase']==1 and r['first_correct_lookahead_local_L']==1 for r in report2['rows']))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);torch.save(model.state_dict(),root/'model.pt');(root/'model_config.json').write_text(json.dumps(dict(arch=arch)))
            answer=''.join(str((r*3+r//3+c)%9+1) for r in range(9) for c in range(9))
            (root/'test.csv').write_text('question,answer\n'+'.'*81+','+answer+'\n')
            args=['--checkpoint',str(root/'model.pt'),'--dataset',str(root/'test.csv'),'--device','cpu','--h-steps','2','--seed','1794200000','--output-dir',str(root/'out')]
            main(args+['--prepare-only']);main(args)
            m=json.loads((root/'out/manifest.json').read_text());self.assertEqual(m['board_ids'],[0]);self.assertEqual(m['native_cycle_bitwise_checks'],1)
            with self.assertRaises(FileExistsError):main(args)
            with self.assertRaises(ValueError):main(args+['--seed','1794200001','--prepare-only'])
