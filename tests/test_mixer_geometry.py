import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from hrm.analyse.dynamics.mixer.geometry import prepare_arrays, run
from hrm.analyse.dynamics.mixer.replay_selected import analysis_h_steps, replay
from hrm.analyse.dynamics.mixer.screen import main as screen
from hrm.analyse.dynamics.jacobian.models import HRM, analysis_architecture


class MixerGeometryTests(unittest.TestCase):
    def test_original_plane_and_splits(self):
        torch.manual_seed(40)
        states=torch.randn(3,33,16);contexts=torch.randn(3,16)
        values,audit=prepare_arrays(states,contexts,1794195000,5,12)
        # Independent original plot construction, including its asymmetric origins.
        torch.manual_seed(1794195000)
        tail=states[1,-12:];mean=tail.mean(0)
        _,_,v=torch.pca_lowrank(tail-mean,q=6,center=False,niter=6)
        vy=states[-1,-1]-mean;vy-=v[:,:2]@(v[:,:2].T@vy);vy/=vy.norm()
        means=states[:,-12:].mean(1);means-=means.mean(0);means-=torch.outer(means@vy,vy)
        _,_,vh=torch.linalg.svd(means,full_matrices=False);vx=vh[0]
        if (means@vx)@torch.arange(3,dtype=means.dtype)<0:vx=-vx
        L=states.flatten(0,1)
        q=torch.stack([(L-states[0,0])@vx,(L-mean)@vy],1)/4
        np.testing.assert_array_equal(values['q'],q.numpy())
        np.testing.assert_allclose(values['B'].T@values['B'],np.eye(2),atol=1e-6)
        np.testing.assert_allclose(values['B'].T@values['L_V'],0,atol=1e-6)
        self.assertLess(audit['coordinate_parity_max_abs'],1e-5)
        self.assertAlmostEqual(values['weights'].sum(),1)
        self.assertEqual(set(values['split']),{0,1,2})
        self.assertTrue((values['split'][(values['step']==0)|(values['step']==32)]==0).all())
        self.assertEqual(analysis_h_steps(dict(h_steps=8,selected=dict(first_correct_H=6))),6)
        self.assertEqual(analysis_h_steps(dict(h_steps=8,selected=dict(case=1))),8)
        with self.assertRaisesRegex(ValueError,'Degenerate'):
            prepare_arrays(torch.zeros(3,33,16),contexts,1,5,12)

    def test_fresh_screen_replay_geometry(self):
        config=dict(transformer=dict(hidden_size=8,intermediate_size=16,head_dim=4,norm_eps=1e-6,
             rope_theta=10000.,forward_dtype='bfloat16',is_mlp_mixer=True,mlp_mixer_intermediate_size=8),
             arch=dict(name='hrm.models.hrm@HRM',num_layers=1,H_cycles=2,L_cycles=6,bptt=True))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cp=root/'model.pt';src=root/'screen';sel=root/'selected';out=root/'geometry'
            torch.manual_seed(45);model=HRM(analysis_architecture(config))
            torch.save(model.state_dict(),cp);(root/'model_config.json').write_text(json.dumps(config))
            data=root/'test.csv';data.write_text('question,answer\n'+'.'*81+','+'123456789'*9+'\n')
            screen(['--checkpoint',str(cp),'--dataset',str(data),'--output-dir',str(src),'--seed','1794195000',
                    '--boards','1','--device','cpu','--l-steps','20','--h-steps','3','--tail-steps','4'])
            replay(src,sel,case=0)
            report=run(sel,out,seed=1794195001,device='cpu',normal_rank=4,tail_size=12)
            self.assertEqual(report['status'],'complete')
            with np.load(out/'prepared.npz') as prepared:
                self.assertEqual(prepared['q'].shape,(63,2))
                self.assertEqual(set(prepared['split']),{0,1,2})
            from hrm.analyse.dynamics.mixer.probe import run as fit_probe
            from hrm.analyse.dynamics.mixer.landscape import run as fit_landscape, ranges
            from hrm.analyse.dynamics.constraint_landscape.global_fit import load_global
            fit_probe(sel,root/'probe',device='cpu',batch_size=4,block_size=1)
            result=fit_landscape(out,root/'probe',sel,root/'landscape',seed=1794195002,
                                 device='cpu',centers=4,global_resolution=5,local_resolution=3,batch_size=4,tail_size=12)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['selection_fit']['n_train'],57)
            self.assertEqual(result['selection_fit']['n_validation'],3)
            self.assertEqual(result['display_refit']['n_train'],63)
            self.assertTrue(result['display_refit']['selected_refit'])
            with np.load(out/'prepared.npz') as prepared,np.load(root/'probe'/'probe.npz') as fitted:
                regions=ranges(prepared['q'],prepared['phase'],12)
                model=load_global(root/'landscape'/'kernel_model.pt',device='cpu')
                for name,shape in [('global',(5,5)),('H1',(3,3)),('H2',(3,3)),('H3',(3,3))]:
                    with np.load(root/'landscape'/f'grid_{name}.npz') as grid:
                        self.assertEqual(grid['probe'].shape,shape)
                        np.testing.assert_allclose([grid['x'].min(),grid['y'].min()],regions[name][0])
                        qi=np.c_[grid['x'].ravel(),grid['y'].ravel()];c=model.predict(qi)
                        z=torch.tensor(prepared['L_base'])+(82*8)**.5*(torch.tensor(qi,dtype=torch.float32)@torch.tensor(prepared['B']).T+
                           torch.tensor(c[:,:prepared['L_V'].shape[1]])@torch.tensor(prepared['L_V']).T)
                        expected=(z.reshape(-1,82,8).bfloat16().float().mean(1).double()@torch.tensor(fitted['weight'])+float(fitted['bias'])).numpy()
                        np.testing.assert_allclose(grid['probe'].ravel(),expected,rtol=1e-5,atol=1e-5)
            with self.assertRaises(FileExistsError):
                fit_landscape(out,root/'probe',sel,root/'landscape',device='cpu')
            with self.assertRaises(FileExistsError):run(sel,out,device='cpu')
            path=sel/'H001_case0_bits.npy';path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'Trajectory changed'):run(sel,root/'bad',device='cpu')


if __name__=='__main__':unittest.main()
