import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hrm.analyse.dynamics.jacobian.models import HRM, analysis_architecture
from hrm.analyse.dynamics.mixer.screen import main as screen
from hrm.analyse.dynamics.mixer.replay_selected import replay
from hrm.analyse.dynamics.mixer.probe import count, temporal_split, fit, run


class MixerProbeTests(unittest.TestCase):
    def test_time_split_and_train_only_fit(self):
        split = temporal_split(np.arange(4097), 4096)
        self.assertEqual(split[4096], split[4095])
        self.assertEqual(split[3*128], 1); self.assertEqual(split[4*128], 2)
        self.assertEqual(np.bincount(split).tolist(), [2561, 768, 768])
        f=np.random.default_rng(19).normal(size=(15,3)); t=f@np.array([1.,2.,3.])+4
        s=np.tile([0,0,0,1,2],3)
        w,b,p,_=fit(f,t,s)
        np.testing.assert_allclose(w,[1,2,3],atol=1e-12);self.assertAlmostEqual(b,4)
        altered=t.copy();altered[s!=0]+=1000
        w2,b2,_,_=fit(f,altered,s)
        np.testing.assert_array_equal(w,w2);self.assertEqual(b,b2)
        with self.assertRaisesRegex(ValueError,'nonempty'):fit(f,t,np.zeros(15))
        logits=torch.zeros(1,82,10);logits[:,:,1]=1
        self.assertEqual(count(logits).item(),216)
        logits.zero_();logits[:,:,0]=1
        self.assertEqual(count(logits).item(),0)

    def test_bf16_screen_replay_probe(self):
        config=dict(transformer=dict(hidden_size=8,intermediate_size=16,head_dim=4,
            norm_eps=1e-6,rope_theta=10000.,forward_dtype='bfloat16',is_mlp_mixer=True,
            mlp_mixer_intermediate_size=8),arch=dict(name='hrm.models.hrm@HRM',num_layers=1,H_cycles=2,L_cycles=6,bptt=True))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cp=root/'model.pt';src=root/'screen';sel=root/'selected';out=root/'probe'
            model=HRM(analysis_architecture(config)).eval()
            torch.save(model.state_dict(),cp);(root/'model_config.json').write_text(json.dumps(config))
            data=root/'test.csv';data.write_text('question,answer\n'+'.'*81+','+'123456789'*9+'\n')
            screen(['--checkpoint',str(cp),'--dataset',str(data),'--output-dir',str(src),'--seed','1794194000',
                    '--boards','1','--device','cpu','--l-steps','5','--h-steps','2','--tail-steps','2'])
            replay(src,sel,case=0)
            report=run(sel,out,device='cpu',batch_size=2,block_size=1)
            self.assertEqual(report['status'],'complete')
            self.assertEqual([report['metrics'][k]['n'] for k in ['train','validation','test']],[6,2,4])
            with np.load(out/'probe.npz') as saved:
                features=saved['features'];targets=saved['target']
            expected_f=[];expected_t=[]
            with torch.inference_mode():
                for h in [1,2]:
                    raw=np.load(sel/f'H{h:03d}_case0_bits.npy')
                    states=torch.from_numpy(raw.copy()).view(torch.bfloat16)
                    context=torch.load(src/f'H{h:03d}.pt',weights_only=True)['H_before'][0]
                    for state in states:
                        expected_f.append(state.float().mean(0).numpy())
                        board=model.lm_head(model.H_level(context+state[None])).argmax(-1)[0,1:].reshape(9,9).numpy()
                        units=list(board)+list(board.T)+[board[i:i+3,j:j+3].ravel() for i in (0,3,6) for j in (0,3,6)]
                        expected_t.append(sum(max(int((unit==digit).sum())-1,0) for unit in units for digit in range(1,10)))
            np.testing.assert_array_equal(features,np.array(expected_f))
            np.testing.assert_array_equal(targets,np.array(expected_t))
            with self.assertRaises(FileExistsError):run(sel,out,block_size=1)
            with self.assertRaisesRegex(ValueError,'too short'):run(sel,root/'short')
            path=sel/'H001_case0_bits.npy';path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'Trajectory changed'):run(sel,root/'bad',block_size=1)


if __name__=='__main__':unittest.main()
