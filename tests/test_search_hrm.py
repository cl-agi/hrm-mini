import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from hrm.models.hrm import HRM
from hrm.analyse.dynamics.search_baselines.run_hrm import trace,distances,main


class HRMSearchTests(unittest.TestCase):
    def test_micro_trace_matches_native_and_distance(self):
        arch=dict(seq_len=82,vocab_size=10,hidden_size=8,intermediate_size=16,head_dim=4,num_layers=1,
            norm_eps=1e-6,rope_theta=10000.,forward_dtype='bfloat16',H_cycles=2,L_cycles=6,bptt=True)
        model=HRM(arch).eval();x=torch.zeros(1,82,dtype=torch.long)
        r=trace(model,x,2);self.assertEqual(len(r['operations']),29)
        self.assertEqual(r['operations'].tolist(),['initial']+(['L']*6+['H'])*4)
        with torch.no_grad():
            carry=model.initial_carry
            for k in range(2):
                carry,logits=model(carry,x)
                np.testing.assert_array_equal(r['zH'][(k+1)*14],carry['z_H'][0].float().numpy())
                np.testing.assert_array_equal(r['zL'][(k+1)*14],carry['z_L'][0].float().numpy())
                np.testing.assert_array_equal(r['predictions'][(k+1)*14],(logits[0,1:,1:].argmax(-1)+1).numpy())
        d,scale=distances(r['zH'],r['zL']);z=(r['zH']+r['zL']).astype(np.float64)
        direct=np.linalg.norm((z[:,None]-z[None,:]).reshape(29,29,-1),axis=-1)
        np.testing.assert_allclose(d,direct/direct.max(),atol=1e-7);self.assertAlmostEqual(scale,direct.max(),6)
        zero,s=distances(r['zH'][:1],r['zL'][:1]);self.assertEqual(s,0);self.assertEqual(zero[0,0],0)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';data.mkdir();cp=root/'model.pt';torch.save(model.state_dict(),cp)
            (root/'model_config.json').write_text(json.dumps(dict(arch=dict(arch,name='hrm.models.hrm@HRM'))))
            answer=''.join(str((i*3+i//3+j)%9+1) for i in range(9) for j in range(9))
            (data/'test.csv').write_text('question,answer\n'+'.'*81+','+answer+'\n')
            out=root/'out';args=['--checkpoint',str(cp),'--dataset',str(data),'--board-id','0','--cycles','2','--device','cpu','--output-dir',str(out)]
            main(args+['--prepare-only']);main(args)
            report=json.loads((out/'manifest.json').read_text());self.assertTrue(report['native_forward_parity']);self.assertEqual(report['total_states'],29)
            self.assertFalse(report['window_ends_at_solution']);self.assertIsNone(report['first_success'])
            with self.assertRaises(FileExistsError):main(args)
