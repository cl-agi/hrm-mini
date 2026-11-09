import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hrm.analyse.dynamics.history_distance.eval_h_step_exploration import main, metrics
from hrm.analyse.dynamics.jacobian.models import HRM, analysis_architecture


class HistoryDistanceTests(unittest.TestCase):
    def test_common_history_and_radial_matching(self):
        origin=torch.tensor([[1.,2.]])
        a=torch.tensor([[3.,2.]])
        b=torch.tensor([[1.,3.]])
        history=torch.tensor([[[0.,0.],[1.,2.],[1.,1.]]])
        result=metrics(a,b,origin,history,torch.tensor([.75]))
        radius=min(np.sqrt(2),np.sqrt(.5))
        for label,query in [('H',a.numpy()),('L',b.numpy())]:
            step=np.sqrt(np.mean((query-origin.numpy())**2,axis=1))
            dist=np.sqrt(np.mean((query[:,None]-history.numpy())**2,axis=-1))
            matched=origin.numpy()+(query-origin.numpy())*(radius/step[:,None])
            matched_dist=np.sqrt(np.mean((matched[:,None]-history.numpy())**2,axis=-1))
            np.testing.assert_allclose(result[label+'_nn'],dist.min(1))
            np.testing.assert_allclose(result[label+'_knn5'],dist.mean(1),rtol=1e-6)
            np.testing.assert_allclose(result[label+'_matched_nn'],matched_dist.min(1),rtol=1e-6)
            np.testing.assert_allclose(result[label+'_occupancy_1'],(dist<=.75).mean(1),rtol=1e-6)

    def test_fresh_entry_and_native_carry(self):
        config=dict(arch=dict(name='hrm.models.hrm@HRM',seq_len=82,vocab_size=10,hidden_size=8,
            intermediate_size=16,head_dim=4,num_layers=1,norm_eps=1e-6,rope_theta=10000.,
            forward_dtype='bfloat16',H_cycles=2,L_cycles=6,bptt=True))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cp=root/'model.pt';model=HRM(analysis_architecture(config));torch.save(model.state_dict(),cp)
            (root/'model_config.json').write_text(json.dumps(config))
            answer=''.join(str((r*3+r//3+c)%9+1) for r in range(9) for c in range(9))
            data=root/'test.csv';pd.DataFrame([dict(question='.'*81,answer=answer),dict(question='1'+'.'*80,answer=answer)]).to_csv(data,index=False)
            out=root/'records';args=['--checkpoint',str(cp),'--dataset',str(data),'--output-dir',str(out),
                '--seed','1794196000','--boards','2','--h-cycles','2','--l-steps','2','6','--batch-size','2','--device','cpu','--bootstrap-repeats','10']
            main(args+['--prepare-only']);main(args)
            report=json.loads((out/'metadata.json').read_text())
            self.assertTrue(report['native_fp32_parity']);self.assertEqual(report['boundary_rows'],16)
            self.assertEqual(report['performance_rows'],8)
            boundaries=pd.read_csv(out/'boundaries.csv');self.assertTrue(np.isfinite(boundaries.select_dtypes('number')).all().all())
            summary=pd.read_csv(out/'summary.csv')
            self.assertFalse(((summary.L_steps==6)&(summary.window=='equal_compute')).any())
            # Full independent native forward must equal two scheduled H6L phases,
            # despite all extra diagnostic calls in the recorded run.
            model.float();model.embed.cast_to=torch.float32;model.eval()
            with np.load(out/'cohort.npz') as cohort,torch.no_grad():
                x=torch.tensor(cohort['inputs']);y=torch.tensor(cohort['targets']);e=model.embed(x)
                carry,logits=model(dict(z_H=model.zH_init.expand_as(e).clone(),z_L=model.zL_init.expand_as(e).clone()),x)
                digest=hashlib.sha256(carry['z_H'].numpy().tobytes()+carry['z_L'].numpy().tobytes()).hexdigest()
                self.assertEqual(next(r['sha256'] for r in report['final_states'] if r['L_steps']==6),digest)
                exact=(logits.argmax(-1)[:,1:]==y[:,1:]).all(1).numpy()
                performance=pd.read_csv(out/'performance.csv');observed=performance[(performance.L_steps==6)&(performance.H_cycle==2)]
                np.testing.assert_array_equal(observed.board_id,cohort['ids']);np.testing.assert_array_equal(observed.exact,exact)
            with self.assertRaises(FileExistsError):main(args)
            with self.assertRaisesRegex(ValueError,'Changed'):main(args+['--seed','1794196001','--prepare-only'])


if __name__=='__main__':unittest.main()
