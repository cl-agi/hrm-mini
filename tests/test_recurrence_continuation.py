import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from hrm.models.hrm import HRM
from hrm.analyse.dynamics.perturbation_recovery.continuation import recurrence,continue_case,read_screen,main


class RecurrenceContinuationTests(unittest.TestCase):
    def test_geometry_direct_difference_and_zero(self):
        z=np.random.default_rng(1794201000).normal(size=(17,2,3))
        report,dist=recurrence(z,8,2)
        direct=np.sqrt(((z[10:,None]-z[None,:8])**2).mean((2,3))).min(1)
        np.testing.assert_allclose(dist,direct,rtol=1e-12,atol=1e-12)
        radius=np.sqrt(((z[:8]-z[:8].mean(0))**2).mean())
        self.assertAlmostEqual(report['nearest_q95_over_radius'],np.quantile(direct,.95)/radius)
        report,_=recurrence(np.zeros_like(z),8,2);self.assertIsNone(report['nearest_q95_over_radius'])

    def test_saved_carry_chain_and_selection(self):
        from hrm.analyse.sudoku.continuous_inference.runner_snapshot import main as screen,parse_args
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);arch=dict(name='hrm.models.hrm@HRM',seq_len=82,vocab_size=10,hidden_size=8,intermediate_size=16,head_dim=4,num_layers=1,norm_eps=1e-6,rope_theta=10000.,forward_dtype='bfloat16',H_cycles=2,L_cycles=6,bptt=True)
            torch.manual_seed(1794201000);model=HRM(arch).eval().requires_grad_(False)
            torch.save(model.state_dict(),r/'model.pt');(r/'model_config.json').write_text(json.dumps(dict(arch=arch)))
            answer=''.join(str((i*3+i//3+j)%9+1) for i in range(9) for j in range(9))
            (r/'test.csv').write_text('question,answer\n'+('.'*81+','+answer+'\n')*3)
            screen(parse_args([str(r/'model.pt'),'--dataset',str(r/'test.csv'),'--seed','1794201000','--device','cpu','--batch-size','1','--max-examples','3','--max-cycles','2','--save-final-carry','--output',str(r/'screen.json')]))
            meta,raw,saved,chosen,n,_=read_screen(r/'screen.json',2)
            self.assertEqual([int(raw['indices'][i]) for i in chosen],[0,1]);self.assertEqual(n,3)
            i=chosen[0];x=torch.tensor(raw['inputs'][i:i+1]);y=torch.tensor(raw['targets'][i:i+1]);carry={k:v[i:i+1] for k,v in saved['carry'].items()}
            arrays,report=continue_case(model,x,y,carry,12,9)
            with torch.inference_mode():
                h=carry['z_H'].clone();l=carry['z_L'].clone();f=l.clone();emb=model.embed(x);hs=[h.float().numpy()[0]];fs=[f.float().numpy()[0]]
                for _ in range(12):
                    for _ in range(6):l=model.L_level(l+h+emb)
                    h=model.H_level(h+l);f=model.L_level(f+carry['z_H']+emb)
                    hs.append(h.float().numpy()[0]);fs.append(f.float().numpy()[0])
            np.testing.assert_array_equal(arrays['H'],hs[-9:]);np.testing.assert_array_equal(arrays['fixed_H_L'],fs[-9:]);self.assertEqual(report['native_cycle_bitwise_checks'],6)
            args=['--records',str(r/'screen.json'),'--checkpoint',str(r/'model.pt'),'--output-dir',str(r/'out'),'--cases','2','--steps','12','--tail','9','--reference','4','--gap','1','--device','cpu']
            main(args);out=json.loads((r/'out/manifest.json').read_text());self.assertEqual(out['selected_ids'],[0,1]);self.assertFalse(out['reference_schedule'])
            with self.assertRaises(FileExistsError):main(args)
            # A solved-then-reverted trace must be excluded even if its last prediction is wrong.
            raw['exact_board81'][chosen[0],0]=True;np.savez_compressed(r/'screen.npz',**raw)
            _,raw2,_,chosen2,n,_=read_screen(r/'screen.json',2)
            self.assertEqual(n,2);self.assertNotIn(0,raw2['indices'][chosen2].tolist())
            saved['indices']=saved['indices'].roll(1);torch.save(saved,r/'screen.carry.pt')
            with self.assertRaisesRegex(ValueError,'ordering'):read_screen(r/'screen.json',2)
