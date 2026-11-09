import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from hrm.models.ablations import HRM
from hrm.analyse.dynamics.old_h_recovery.run import main, select_split
from hrm.analyse.dynamics.jacobian.models import analysis_architecture
from hrm.analyse.dynamics.jacobian.data import SudokuRecord


class OldHFeaturesTests(unittest.TestCase):
    def test_decoder_selection_does_not_use_test_targets(self):
        from hrm.analyse.sudoku.core.old_h_recovery import ridge, fit_mlp, metrics
        g=torch.Generator().manual_seed(1794199000)
        x=torch.randn(12,3,generator=g);y=torch.randn(12,4,generator=g)
        changed=y.clone();changed[8:]+=100
        w,selection=ridge(x,y,6,2);other,other_selection=ridge(x,changed,6,2)
        self.assertTrue(torch.equal(w,other));self.assertEqual(selection,other_selection)
        _,weights,history=fit_mlp(x,y,w,6,2,1794199000,1)
        _,other_weights,other_history=fit_mlp(x,changed,w,6,2,1794199000,1)
        self.assertEqual(history,other_history)
        for key in weights:self.assertTrue(torch.equal(weights[key],other_weights[key]))
        result,_=metrics(torch.zeros(4,3),torch.zeros(4,3),torch.zeros(4,3),torch.ones(4,dtype=torch.bool),2,2,1794199000,10)
        self.assertIsNone(result['r2']);self.assertIsNone(result['given_r2']);self.assertIsNone(result['conditional_puzzle_r2_ci95'])

    def test_features_and_link_intervention(self):
        config=dict(arch=dict(name='hrm.models.ablations@HRM',seq_len=82,vocab_size=10,hidden_size=8,
            intermediate_size=16,head_dim=4,num_layers=1,norm_eps=1e-6,rope_theta=10000.,
            forward_dtype='bfloat16',H_cycles=2,L_cycles=6,bptt=True,hh_link=True))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';data.mkdir();cp=root/'model.pt'
            model=HRM(analysis_architecture(config));torch.save(model.state_dict(),cp)
            (root/'model_config.json').write_text(json.dumps(config))
            answer=''.join(str((r*3+r//3+c)%9+1) for r in range(9) for c in range(9))
            pd.DataFrame([dict(question='9'+'.'*80,answer=answer)]).to_csv(data/'train.csv',index=False)
            rows=[dict(question=str(i)+'.'*80,answer=answer) for i in range(1,6)]
            pd.DataFrame(rows).to_csv(data/'test.csv',index=False)
            args=['--checkpoint',str(cp),'--dataset',str(data),'--seed','1794199000','--train-count','2',
                '--val-count','1','--test-count','2','--tokens','2','--steps','1','2','--batch-size','2','--device','cpu']
            splits=[]
            for link in ['native','deleted']:
                out=root/link;main(args+['--hh-link',link,'--output-dir',str(out),'--prepare-only'])
                main(args+['--hh-link',link,'--output-dir',str(out)])
                report=json.loads((out/'manifest.json').read_text());self.assertTrue(report['native_forward_parity'])
                self.assertEqual(report['no_hh_identity_checked'],link=='deleted')
                saved=torch.load(out/'features.pt',weights_only=True);self.assertEqual(saved['1/h'].shape,(5,2,8))
                splits.append(saved['indices']);self.assertEqual(len(set(saved['indices'].tolist())),5)
                # Independent native cycle supplies the old H at update index 2.
                m=HRM(analysis_architecture(config));m.load_state_dict(model.state_dict());m.hh_link=link!='deleted';m.eval()
                x=torch.tensor([[0]+[int(rows[int(i)]['question'][0])]+[0]*80 for i in saved['indices']])
                with torch.no_grad():
                    e=m.embed(x);carry,_=m(dict(z_H=m.zH_init.expand_as(e).clone(),z_L=m.zL_init.expand_as(e).clone()),x)
                positions=saved['positions']+1
                torch.testing.assert_close(saved['2/h'],carry['z_H'][torch.arange(5)[:,None],positions],rtol=0,atol=0)
                from hrm.analyse.dynamics.old_h_recovery.fit import main as fit_main
                from hrm.analyse.sudoku.core.old_h_recovery import position_mean
                decoded=root/(link+'_decoders')
                fit_main(['--records',str(out),'--output-dir',str(decoded),'--device','cpu','--epochs','1','--bootstrap-repeats','10','--seed','1794199001'])
                results=json.loads((decoded/'results.json').read_text());self.assertEqual(len(results),28)
                decoder=torch.load(decoded/'probe_1_next_l.pt',weights_only=True)
                pos=saved['positions'].flatten();target=saved['1/h'].flatten(0,1).float()
                torch.testing.assert_close(decoder['ymean'],position_mean(target[:4],pos[:4]),rtol=0,atol=0)
                raw=saved['1/next_l'].flatten(0,1).float();test=slice(6,None)
                xx=(raw-decoder['xmean'][pos])/decoder['xscale'];mu=decoder['ymean'][pos]
                net=torch.nn.Sequential(torch.nn.Linear(8,256),torch.nn.GELU(),torch.nn.Linear(256,8))
                net.load_state_dict(decoder['mlp']);net.eval()
                with torch.no_grad():
                    predicted=xx[test]@decoder['ridge']+net(xx[test])+mu[test]
                    expected_r2=float(1-(predicted-target[test]).square().sum()/(target[test]-mu[test]).square().sum())
                row=next(r for r in results if r['step']==1 and r['probe']=='next_l_mlp')
                self.assertAlmostEqual(expected_r2,row['r2'],places=5)
                self.assertEqual(decoder['linear_selection']['validation_mse'],min(r['validation_mse'] for r in decoder['linear_selection']['grid']))
                self.assertEqual(decoder['mlp_selection']['validation_mse'],min(r['validation_mse'] for r in decoder['mlp_selection']['history']))
                self.assertTrue((decoded/'probe_1_next_l_shuffled.pt').exists())
                from hrm.analyse.dynamics.old_h_recovery.audit import main as audit_main
                from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
                audited=root/(link+'_audit')
                audit_args=['--records',str(decoded),'--feature-records',str(out),'--output-dir',str(audited)]
                audit_main(audit_args)
                check=json.loads((audited/'audit.json').read_text());self.assertEqual(len(check['checks']),28)
                with self.assertRaises(FileExistsError):audit_main(audit_args)
                original=(decoded/'results.json').read_text();original_manifest=(decoded/'manifest.json').read_text()
                altered=json.loads(original);altered[0]['r2']+=.5
                (decoded/'results.json').write_text(json.dumps(altered))
                manifest=json.loads(original_manifest);manifest['artifacts']['results.json']=sha(decoded/'results.json')
                (decoded/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'Numerical mismatch'):
                    audit_main(['--records',str(decoded),'--feature-records',str(out),'--output-dir',str(root/(link+'_bad_audit'))])
                (decoded/'results.json').write_text(original);(decoded/'manifest.json').write_text(original_manifest)

                with self.assertRaises(FileExistsError):main(args+['--hh-link',link,'--output-dir',str(out)])
            self.assertTrue(torch.equal(*splits))
            record=SudokuRecord(0,'.'*81,answer)
            with self.assertRaisesRegex(ValueError,'overlaps'):select_split([record],[record],1,[1,0,0],1)
            with self.assertRaisesRegex(ValueError,'distinct'):select_split([], [record,record],1,[2,0,0],1)


if __name__=='__main__':unittest.main()
