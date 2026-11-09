import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from hrm.analyse.dynamics.mixer.screen import main, rank_candidates
from hrm.analyse.dynamics.jacobian.models import HRM, analysis_architecture


class MixerScreenTests(unittest.TestCase):
    def test_frozen_cohort_and_original_recurrence(self):
        config = dict(transformer=dict(hidden_size=8, intermediate_size=16, head_dim=4,
            norm_eps=1e-6, rope_theta=10000., forward_dtype='bfloat16', is_mlp_mixer=True,
            mlp_mixer_intermediate_size=8), arch=dict(name='hrm.models.hrm@HRM', num_layers=1,
            H_cycles=2, L_cycles=6, bptt=True))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cp=root/'model.pt'; data=root/'test.csv'; out=root/'screen'
            model=HRM(analysis_architecture(config)).eval()
            torch.save(model.state_dict(),cp)
            (root/'model_config.json').write_text(json.dumps(config))
            data.write_text('question,answer\n'+'.'*81+','+'123456789'*9+'\n'+
                            '1'+'.'*80+','+'123456789'*9+'\n')
            args=['--checkpoint',str(cp),'--dataset',str(data),'--output-dir',str(out),
                  '--boards','2','--device','cpu','--l-steps','4','--h-steps','3','--tail-steps','3']
            with patch('time.time',return_value=1794193000):main(args+['--prepare-only'])
            main(args+['--prepare-only']); main(args)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['seed'],1794193000)
            self.assertEqual(manifest['status'],'complete')
            with np.load(out/'cohort.npz') as cohort:
                x=torch.tensor(cohort['inputs']); y=torch.tensor(cohort['targets'])
            with torch.inference_mode():
                emb=model.embed(x);h=model.zH_init.expand_as(emb).clone();l=model.zL_init.expand_as(emb).clone()
                for hc in range(1,4):
                    before=h.clone();tail=[]
                    for step in range(1,5):
                        l=model.L_level(l+h+emb)
                        if step>1:tail.append(l.clone())
                    h=model.H_level(h+l)
                    saved=torch.load(out/f'H{hc:03d}.pt',weights_only=True)
                    for k,v in dict(H_before=before,H_after=h,L_after=l,tail=torch.stack(tail),predictions=model.lm_head(h).argmax(-1)).items():
                        self.assertTrue(torch.equal(saved[k],v),k)
            from hrm.analyse.dynamics.mixer.replay_selected import replay
            from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha
            with self.assertRaisesRegex(ValueError, 'No eligible case'):
                replay(out, root/'automatic')
            selected = int(manifest['board_ids'][0])
            result = replay(out, root/'replay', case=selected)
            self.assertEqual(result['status'], 'complete')
            self.assertIn('diagnostic', result['selection'])
            with torch.inference_mode():
                emb=model.embed(x);h=model.zH_init.expand_as(emb).clone();l=model.zL_init.expand_as(emb).clone()
                for hc in range(1,4):
                    trace=np.load(root/'replay'/f'H{hc:03d}_case{selected}_bits.npy')
                    self.assertTrue(np.array_equal(trace[0],l[0].contiguous().view(torch.uint16).numpy()))
                    for step in range(1,5):
                        l=model.L_level(l+h+emb)
                        self.assertTrue(np.array_equal(trace[step],l[0].contiguous().view(torch.uint16).numpy()))
                    h=model.H_level(h+l)
            with self.assertRaises(FileExistsError):replay(out, root/'replay', case=selected)
            with self.assertRaisesRegex(ValueError, 'absent'):replay(out, root/'absent', case=999)
            boundary=out/'H001.pt'
            original=boundary.read_bytes()
            boundary.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError, 'record changed'):replay(out, root/'corrupt', case=selected)
            boundary.write_bytes(original)
            bad=torch.load(boundary,weights_only=True);bad['H_after']+=1;torch.save(bad,boundary)
            manifest['boundary_sha256']['H001.pt']=sha(boundary)
            (out/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError,'H_after differs'):
                replay(out, root/'mismatch', case=selected)
            self.assertFalse((root/'mismatch'/'manifest.json').exists())
            with self.assertRaises(FileExistsError):main(args)
            with self.assertRaisesRegex(ValueError,'changed'):main(args+['--seed','1794193001','--prepare-only'])

    def test_archived_ranking_rule(self):
        summaries=[dict(case=i,first_correct_H=first,stable_after_first=stable) for i,first,stable in
                   [(10,4,True),(11,5,True),(12,4,False),(13,7,True),(14,None,False)]]
        rows=[dict(case=i,H_cycle=h,tail_pc123=.9,tail_scatter_rms=.03)
              for i in range(10,15) for h in range(1,9)]
        ranked=rank_candidates(summaries,rows)
        self.assertEqual([r['case'] for r in ranked],[11,10])
        self.assertEqual(ranked[0]['clear_pre_solution_cycles'],3)


if __name__=='__main__':unittest.main()
