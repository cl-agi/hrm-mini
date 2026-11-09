import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from hrm.analyse.dynamics.output_sensitivity import h_l_joint_transition_jacobian as local
from hrm.analyse.dynamics.output_sensitivity import h_l_natural_taylor_prediction as taylor
from hrm.analyse.dynamics.output_sensitivity.local_analysis import main
from hrm.analyse.dynamics.jacobian.models import HRM


class Scale(nn.Module):
    def __init__(self, value):
        super().__init__(); self.value = value
    def forward(self, x):
        return self.value*x


class AffineHRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 2)
        with torch.no_grad(): self.embed.weight.fill_(0.01)
        self.initial_carry = {'z_H':torch.zeros(2), 'z_L':torch.zeros(2)}
        self.L_level = Scale(.1); self.H_level = Scale(.2); self.lm_head = Scale(1.)
        self.requires_grad_(False)


class LocalJacobianTests(unittest.TestCase):
    def test_four_blocks_match_analytic_affine_jacobian(self):
        rows, error = local.run_batch(model=AffineHRM(), x=torch.zeros(1,82,dtype=torch.long), board_ids=[7], probes=1, generator=torch.Generator().manual_seed(100))
        a=.1**6; b=sum(.1**k for k in range(1,7))
        expected={'j_h_from_h_gain_sq':(.2*(1+b))**2,'j_l_from_h_gain_sq':b**2,'j_h_from_l_gain_sq':(.2*a)**2,'j_l_from_l_gain_sq':a**2}
        self.assertEqual(len(rows),32); self.assertEqual(error,0.)
        for row in rows:
            for k,v in expected.items():self.assertAlmostEqual(row[k]/v,1.,places=5)
        secant, rollout, error=taylor.run_batch(AffineHRM(),torch.zeros(1,82,dtype=torch.long),[7],[0],2)
        self.assertEqual(len(secant),31*8); self.assertEqual(len(rollout),9)
        self.assertEqual(error,0.)
        with tempfile.TemporaryDirectory() as d:
            taylor.atomic_csv(Path(d)/'rollout.csv',rollout)
            with (Path(d)/'rollout.csv').open() as f:
                self.assertIn('argmax_cell_match_fraction',next(csv.reader(f)))

    def test_tiny_checkpoint_to_records_and_frozen_prepare(self):
        torch.set_num_threads(1)
        arch=dict(name='hrm.models.hrm@HRM',num_layers=1,hidden_size=8,intermediate_size=16,head_dim=4,norm_eps=1e-6,rope_theta=10000.,H_cycles=2,L_cycles=6,bptt=True,forward_dtype='float32',is_causal=False,vocab_size=10,seq_len=82)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);checkpoint=root/'model.pt';config=root/'model_config.json';data=root/'test.csv'
            torch.save(HRM(arch).state_dict(),checkpoint);config.write_text(json.dumps({'arch':arch}))
            data.write_text('question,answer\n'+'.'*81+','+'123456789'*9+'\n')
            for kind in ['local','taylor']:
                output=root/kind
                args=['--checkpoint',str(checkpoint),'--dataset',str(data),'--output-dir',str(output),'--boards','1','--batch-size','1','--device','cpu','--rollout-sources','0','--max-horizon','2']
                with patch('time.time',return_value=1794188000): main(kind,args+['--prepare-only'])
                # Exercise the documented default, independently of explicit output paths.
                default_args=[v for i,v in enumerate(args) if i not in (4,5)]
                with patch.dict(os.environ, {}, clear=True), patch('hrm.analyse.dynamics.output_sensitivity.local_analysis.repository_root',return_value=root):
                    main(kind,default_args+['--prepare-only'])
                stage='local_jacobian' if kind=='local' else 'taylor'
                self.assertTrue((root/'exp/memory/records'/stage/'run_config.json').is_file())
                first=(output/'run_config.json').read_bytes()
                main(kind,args+['--prepare-only'])
                self.assertEqual(first,(output/'run_config.json').read_bytes())
                main(kind,args)
                manifest=json.loads((output/'manifest.json').read_text())
                self.assertEqual(manifest['status'],'complete');self.assertEqual(manifest['board_ids'],[0]);self.assertEqual(manifest['seed'],1794188000)
                self.assertEqual(manifest['rows'],32 if kind=='local' else 248)
                with self.assertRaises(FileExistsError): main(kind,args)
                with self.assertRaises(ValueError):main(kind,args+['--seed','1794188001','--prepare-only'])
            with patch('hrm.hub.checkpoint',side_effect=AssertionError('HF fallback')):
                # Explicit local data errors are checked independently by the reader.
                from hrm.analyse.dynamics.jacobian.data import read_test_records
                with self.assertRaises(ValueError):read_test_records(root/'missing.csv')

    def test_mixer_nested_config_and_three_audits(self):
        from hrm.analyse.dynamics.jacobian.models import analysis_architecture, build_model
        from hrm.analyse.dynamics.mixer.audit import main as mixer_main
        torch.set_num_threads(1)
        config = dict(transformer=dict(num_layers=4, hidden_size=8, intermediate_size=16,
            head_dim=4, norm_eps=1e-6, rope_theta=10000., forward_dtype='float32',
            is_mlp_mixer=True, mlp_mixer_intermediate_size=8),
            arch=dict(name='hrm.models.hrm@HRM',num_layers=1,H_cycles=2,L_cycles=6,bptt=True,cycles_per_data=16))
        arch=analysis_architecture(config)
        self.assertEqual(arch['num_layers'],1)
        self.assertNotIn('cycles_per_data',arch)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); checkpoint=root/'model.pt'; cfg=root/'model_config.json'; data=root/'test.csv'
            model=HRM(arch); torch.save(model.state_dict(),checkpoint);cfg.write_text(json.dumps(config))
            restored,_=build_model(checkpoint,cfg,torch.device('cpu'))
            for k,v in model.state_dict().items(): self.assertTrue(torch.equal(v,restored.state_dict()[k]))
            # Compare the shared output-triangle kernel to the original H/L audit,
            # including identical per-board/endpoint probe seed derivation.
            from hrm.analyse.dynamics.jacobian.compute import run_batch as shared_triangle
            from hrm.analyse.dynamics.output_sensitivity.h_l_output_triangle_vjp import run_batch as original_triangle
            x=torch.zeros(1,82,dtype=torch.long); y=torch.ones_like(x)
            current,_=shared_triangle(restored,x,y,[7],outer_steps=16,probes=1,random_seed=20260828)
            reference,_=original_triangle(restored,x,y,[7],1,20260828,torch.eye(8),torch.eye(8),None)
            def values(rows):
                return {(r['board_id'],r['source_step'],r['readout_step'],r['source_state'],r['probe']):r['gain_sq'] for r in rows}
            a,b=values(current),values(reference)
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.assertAlmostEqual(a[k],b[k],places=7)
            data.write_text('question,answer\n'+'.'*81+','+'123456789'*9+'\n')
            for kind in ['local','taylor','triangle']:
                output=root/kind
                args=[kind,'--checkpoint',str(checkpoint),'--dataset',str(data),'--output-dir',str(output),
                    '--boards','1','--batch-size','1','--device','cpu','--probes','1','--rollout-sources','0','--max-horizon','2']
                default_args=[v for i,v in enumerate(args) if i not in (5,6)]
                with patch.dict(os.environ, {}, clear=True), patch('hrm.analyse.dynamics.output_sensitivity.local_analysis.repository_root',return_value=root):
                    mixer_main(default_args+['--prepare-only'])
                self.assertTrue((root/'exp/dynamics/records/mixer'/kind/'run_config.json').is_file())
                mixer_main(args+['--prepare-only']); mixer_main(args)
                result=json.loads((output/'manifest.json').read_text())
                self.assertTrue(result['mixer']);self.assertEqual(result['status'],'complete')
                if kind=='triangle':
                    self.assertEqual(result['checks']['per_board_rows_actual'],272)
                    self.assertTrue(result['checks']['all_values_finite'])
                    self.assertEqual(result['checks']['diagonal_gain_sq']['L']['maximum'],0.)
                else:self.assertEqual(result['rows'],32 if kind=='local' else 248)
            wrong=dict(config,transformer=dict(config['transformer'],is_mlp_mixer=False))
            cfg.write_text(json.dumps(wrong))
            with self.assertRaises(SystemExit):mixer_main(['local','--checkpoint',str(checkpoint),'--prepare-only'])
            with patch('hrm.hub.checkpoint', side_effect=AssertionError('must not fetch attention checkpoint')):
                with self.assertRaises(SystemExit):mixer_main(['local','--prepare-only'])

if __name__=='__main__':unittest.main()
