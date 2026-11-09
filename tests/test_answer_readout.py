import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hrm.analyse.dynamics.answer_direction.readout import main, split_data
from hrm.analyse.dynamics.jacobian.models import HRM, analysis_architecture


class AnswerReadoutTests(unittest.TestCase):
    def test_frozen_backbone_split_and_heldout_independence(self):
        config=dict(arch=dict(name='hrm.models.hrm@HRM',seq_len=82,vocab_size=10,hidden_size=8,
          intermediate_size=16,head_dim=4,num_layers=1,norm_eps=1e-6,rope_theta=10000.,
          forward_dtype='bfloat16',H_cycles=2,L_cycles=6,bptt=True))
        base=np.array([((r*3+r//3+c)%9) for r in range(9) for c in range(9)])
        rows=[dict(question=str(i+1)+'.'*80,answer=''.join(map(str,(base+i)%9+1))) for i in range(6)]
        train=pd.DataFrame(rows[:3]);test=pd.DataFrame(rows[3:])
        split=split_data(train,test,1794197000,2,1,2,[rows[3]['question'].replace('.','0')])
        self.assertEqual(split['excluded_test_rows'],[0]);self.assertEqual(set(split['test_rows']),{1,2})
        with self.assertRaisesRegex(ValueError,'overlap'):split_data(train,train,1,1,1,1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cp=root/'model.pt';data=root/'data';data.mkdir()
            model=HRM(analysis_architecture(config));torch.save(model.state_dict(),cp)
            (root/'model_config.json').write_text(json.dumps(config))
            train.to_csv(data/'train.csv',index=False);test.to_csv(data/'test.csv',index=False)
            exclude=root/'exclude.npz';np.savez(exclude,inputs=[[0]+[int(c) for c in rows[3]['question'].replace('.','0')]])
            args=['--checkpoint',str(cp),'--dataset',str(data),'--exclude-cohort',str(exclude),
                  '--seed','1794197000','--train-count','2','--val-count','1','--test-count','2',
                  '--horizon','3','--cells-per-board','2','--steps','2','--half-batch','2','--batch-size','2','--device','cpu']
            out=root/'readout';main(args+['--output-dir',str(out),'--prepare-only']);main(args+['--output-dir',str(out)])
            report=json.loads((out/'manifest.json').read_text());self.assertEqual(report['status'],'complete')
            heads=torch.load(out/'secondary_heads.pt',weights_only=True)
            self.assertEqual(len(heads['heads']),4)
            selected=min([n for n in heads['heads'] if n.startswith('sum')],key=lambda n:heads['heads'][n]['validation']['balanced_ce'])
            self.assertEqual(heads['selected'],selected)
            with np.load(out/'heldout_trajectories.npz') as f:
                self.assertEqual(f['native_pred'].shape,(2,22,81))
                x=torch.tensor(f['input']);emb=model.embed(x);h=model.zH_init.expand_as(emb).clone();l=model.zL_init.expand_as(emb).clone()
                with torch.no_grad():
                    for cycle in range(1,4):
                        for _ in range(6):l=model.L_level(l+h+emb)
                        h=model.H_level(h+l)
                        np.testing.assert_array_equal(f['native_pred'][:,cycle*7],model.lm_head(h).argmax(-1)[:,1:].numpy())
            from hrm.analyse.dynamics.answer_direction.probe_hl_direction_intervention import main as intervene
            branch=root/'intervention'
            intervene(['--records',str(out),'--output-dir',str(branch),'--device','cpu'])
            audit=json.loads((branch/'manifest.json').read_text())
            self.assertEqual(audit['native_boards_exactly_replayed'],6)
            self.assertEqual(audit['selected'],selected)
            # Independently reconstruct the second L6 boundary and its paired branches.
            with np.load(out/'heldout_trajectories.npz') as trajectory, np.load(branch/'branch_logits.npz') as result, torch.no_grad():
                x=torch.tensor(trajectory['input']);e=model.embed(x)
                h=model.zH_init.expand_as(e).clone();l=model.zL_init.expand_as(e).clone()
                for cycle in range(2):
                    for _ in range(6):l=model.L_level(l+h+e)
                    if cycle==0:h=model.H_level(h+l)
                hn=model.H_level(h+l);ln=model.L_level(l+h+e)
                w=heads['heads'][selected]['weight']
                expected={
                    'H_only':torch.nn.functional.linear(hn[:,1:].float()+l[:,1:].float(),w),
                    'L_only':torch.nn.functional.linear(h[:,1:].float()+ln[:,1:].float(),w),
                    'L_then_H':torch.nn.functional.linear(model.H_level(h+ln)[:,1:].float()+ln[:,1:].float(),w)}
                self.assertGreater(audit['states'],0)
                for j,rid in enumerate(result['ids']):
                    i=trajectory['ids'].tolist().index(rid)
                    self.assertEqual(result['cycles'][j],2)
                    for name,value in expected.items():np.testing.assert_array_equal(result[name][j],value[i].numpy())
            with self.assertRaises(FileExistsError):intervene(['--records',str(out),'--output-dir',str(branch),'--device','cpu'])
            from hrm.analyse.dynamics.answer_direction.analyze import main as summarize, paired_summary
            stats=root/'statistics'
            summarize(['--records',str(branch),'--readout-records',str(out),'--output-dir',str(stats),'--seed','1794197001','--bootstrap-repeats','50'])
            summary=pd.read_csv(stats/'summary.csv')
            self.assertEqual(len(summary),12)
            states=pd.read_csv(stats/'per_state.csv')
            self.assertEqual(len(states),audit['states'])
            from hrm.analyse.dynamics.answer_direction.analyze_hl_single_step import main as single_step, conflicts
            diagnostic=root/'single_step'
            single_step(['--records',str(branch),'--readout-records',str(out),'--output-dir',str(diagnostic)])
            check=json.loads((diagnostic/'branch_checks.json').read_text())
            self.assertEqual(check['selected'],selected)
            self.assertEqual(check['matched_board_checks'],4*audit['states'])
            self.assertEqual(check['unique_peer_pairs'],810)
            valid=base+1
            broken=valid.copy();broken[0]=broken[1]
            np.testing.assert_array_equal(conflicts(np.stack([valid,broken,np.zeros(81,int),np.ones(81,int)])),[0,2,0,810])
            diagnostic_table=pd.read_csv(diagnostic/'same_state_single_step.csv')
            self.assertEqual(len(diagnostic_table),audit['states'])
            with self.assertRaises(FileExistsError):
                single_step(['--records',str(branch),'--readout-records',str(out),'--output-dir',str(diagnostic)])
            # Paired bootstrap preserves constant per-puzzle differences exactly.
            paired=paired_summary(np.array([[3.,1.],[5.,3.],[9.,7.]]),1794197001,50)
            self.assertEqual(paired['H_minus_L'],2.)
            self.assertAlmostEqual(paired['ci_low'],2.)
            self.assertAlmostEqual(paired['ci_high'],2.)
            self.assertEqual(paired_summary(np.empty((0,2)),1,50)['puzzles'],0)
            self.assertIsNone(paired_summary(np.empty((0,2)),1,50)['H_mean'])
            # Even a rehashed artifact must not silently omit a paired state.
            from hrm.analyse.dynamics.answer_direction.analyze import sha
            logits_path=branch/'branch_logits.npz';original_logits=logits_path.read_bytes()
            manifest_path=branch/'manifest.json';original_manifest=manifest_path.read_text()
            with np.load(logits_path) as f:short={k:f[k][1:] for k in f.files}
            np.savez_compressed(logits_path,**short)
            altered=json.loads(original_manifest);altered['artifact_sha256']['branch_logits.npz']=sha(logits_path)
            manifest_path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,'missing'):
                summarize(['--records',str(branch),'--readout-records',str(out),'--output-dir',str(root/'missing')])
            logits_path.write_bytes(original_logits);manifest_path.write_text(original_manifest)
            original=(out/'secondary_heads.pt').read_bytes()
            (out/'secondary_heads.pt').write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                intervene(['--records',str(out),'--output-dir',str(root/'bad'),'--device','cpu'])
            (out/'secondary_heads.pt').write_bytes(original)
            # Held-out answers may change metrics, but must not change fitted heads.
            test['answer']=test['answer'].iloc[::-1].to_numpy();test.to_csv(data/'test.csv',index=False)
            second=root/'readout2';main(args+['--output-dir',str(second)])
            other=torch.load(second/'secondary_heads.pt',weights_only=True)
            for name in heads['heads']:self.assertTrue(torch.equal(heads['heads'][name]['weight'],other['heads'][name]['weight']))
            with self.assertRaisesRegex(ValueError,'Changed'):main(args+['--output-dir',str(out),'--prepare-only'])
            with self.assertRaises(FileExistsError):main(args+['--output-dir',str(second)])


if __name__=='__main__':unittest.main()
