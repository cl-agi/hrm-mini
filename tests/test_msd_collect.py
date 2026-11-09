import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from hrm.models.hrm import HRM
from hrm.models.trm import TRM
from hrm.models.rt import RecurrentTransformer
from hrm.analyse.dynamics.msd.collect import states,window_msd,collect,main,build_model,BOUNDARIES
from hrm.analyse.dynamics.msd.analyze import load_curves,analyse


def architecture(kind):
    d=dict(name='hrm.models.hrm@HRM',seq_len=82,vocab_size=10,hidden_size=8,intermediate_size=16,
           head_dim=4,num_layers=1,norm_eps=1e-6,rope_theta=10000.,forward_dtype='bfloat16',H_cycles=2,L_cycles=6,bptt=True)
    if kind=='trm':d['name']='hrm.models.trm@TRM'
    if kind=='hrm_h2l1':d['L_cycles']=1
    if kind=='rt':d.update(name='hrm.models.rt@RecurrentTransformer',cycles=2)
    return d


class MSDCollectionTests(unittest.TestCase):
    def test_clock_matches_native_carry_for_all_architectures(self):
        torch.manual_seed(25)
        for kind in ('hrm','hrm_h2l1','trm','rt'):
            config=architecture(kind);cls=RecurrentTransformer if kind=='rt' else TRM if kind=='trm' else HRM
            model=cls(config).eval().requires_grad_(False)
            inputs=torch.zeros(2,82,dtype=torch.long);inputs[1,2]=3
            cycle=2 if kind=='rt' else 2*config['L_cycles']
            history=list(states(model,inputs,cycle*2,kind))
            with torch.inference_mode():
                carry=model.initial_carry
                for i in (1,2):
                    carry,_=model(carry,inputs)
                    for state,key in ({'rt':'z'} if kind=='rt' else {'h':'z_H','l':'z_L'}).items():
                        np.testing.assert_array_equal(history[i*cycle][state],carry[key].float().numpy())
            if kind in ('hrm','trm'):
                for i in range(1,6):np.testing.assert_array_equal(history[0]['h'],history[i]['h'])

    def test_squared_displacements_and_origin_counts(self):
        t=np.arange(12,dtype=np.float32)[:,None,None,None]
        history=t*np.array([1.,3.],dtype=np.float32)[None,:,None,None]
        for lag in (1,6,11):self.assertAlmostEqual(window_msd(history,lag,chunk=2),5*lag**2)
        with self.assertRaises(ValueError):window_msd(history,12)
        model=HRM(architecture('hrm')).eval().requires_grad_(False);inputs=torch.zeros(3,82,dtype=torch.long)
        inputs[1,1]=1;inputs[2,2]=2;bounds=[0,12,24,36,48];lags=[1,6,9]
        all_states=list(states(model,inputs,48,'hrm'))
        expected={}
        for segment,(a,b) in enumerate(zip(bounds,bounds[1:]),1):
            for state in ('h','l'):
                for lag in lags:
                    expected[state,segment,lag]=np.mean([np.square(all_states[t+lag][state]-all_states[t][state]).mean(dtype=np.float64) for t in range(a,b-lag)])
        batches=[(inputs[:2],inputs[:2],None),(inputs[2:],inputs[2:],None)]
        result,n=collect(model,batches,bounds,lags,'hrm','cpu');self.assertEqual(n,3)
        for key,value in result.items():self.assertAlmostEqual(value,expected[key],places=7)

    def test_local_and_hf_default_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);answer=''.join(str((r*3+r//3+c)%9+1) for r in range(9) for c in range(9))
            (root/'test.csv').write_text('question,answer\n'+'0'*81+','+answer+'\n')
            for kind in ('hrm','trm','hrm_h2l1','rt'):
                arch=architecture(kind);cls=RecurrentTransformer if kind=='rt' else TRM if kind=='trm' else HRM
                torch.save(cls(arch).state_dict(),root/'model.pt');(root/'model_config.json').write_text(json.dumps(dict(arch=arch,seeds=[1794246000])))
                output=root/kind
                args=['--checkpoint',str(root/'model.pt'),'--dataset',str(root/'test.csv'),'--output-dir',str(output),'--device','cpu','--boards','1','--boundaries','0','12','24','36','48']
                main(args+['--prepare-only']);frozen=(output/'run_config.json').read_text();main(args)
                self.assertEqual(frozen,(output/'run_config.json').read_text())
                manifest=json.loads((output/'manifest.json').read_text());self.assertEqual(manifest['condition'],kind)
                report=json.loads((output/'statistics/analysis_manifest.json').read_text());self.assertEqual(report['replicate_unit'],'checkpoint')
                self.assertEqual(report['runs'][0]['metadata']['seed'],'1794246000')
                curves,groups,_=load_curves(output/'rollouts');self.assertTrue(curves)
                with self.assertRaises(ValueError):analyse(groups)
                self.assertFalse(list(output.rglob('*.npz')));self.assertFalse(list(output.rglob('*.png')))
                with self.assertRaises(FileExistsError):main(args)
            # Default checkpoint/data resolver uses the same complete collection path.
            arch=architecture('hrm');torch.save(HRM(arch).state_dict(),root/'model.pt')
            with patch('hrm.hub.checkpoint',return_value=root/'model.pt') as cp,patch('hrm.hub.model_config',return_value=arch),patch('hrm.analyse.dynamics.msd.collect.puzzle_directory',return_value=root):
                main(['--output-dir',str(root/'default'),'--device','cpu','--boards','1','--boundaries','0','12','24','36','48'])
                cp.assert_called_once_with('sudoku',None)
            # Full default window/lag grid connects to all existing fit profiles.
            (root/'model_config.json').write_text(json.dumps(dict(arch=arch)))
            main(['--checkpoint',str(root/'model.pt'),'--dataset',str(root/'test.csv'),'--output-dir',str(root/'full'),'--device','cpu','--boards','1'])
            report=json.loads((root/'full/statistics/analysis_manifest.json').read_text())
            self.assertEqual(report['outputs']['beta_per_seed.csv']['rows'],26)
            self.assertEqual(json.loads(report['runs'][0]['metadata']['segment_boundaries_l_updates']),BOUNDARIES)
