import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from hrm.analyse.sudoku.stable16.inference import iter_inference
from hrm.evaluation.streaming import iter_inference as single_budget
from hrm.analyse.sudoku.stable16.evaluate_distributed import parse_args, evaluate, validate_complete_results
from hrm.models.hrm import HRM


class SettlingModel(nn.Module):
    def __init__(self):
        super().__init__();self.p=nn.Parameter(torch.zeros(()))
        self.initial_carry={'z':torch.zeros(1)}
    def forward(self,carry,x):
        steps=carry['z']+1
        prediction=torch.where(steps[:,:,0]<x,steps[:,:,0].long()%2,torch.ones_like(x))
        return {'z':steps},torch.nn.functional.one_hot(prediction,num_classes=2).float()


class Stable16Tests(unittest.TestCase):
    def test_continuous_budgets_equal_independent_runs_with_refills(self):
        model=SettlingModel()
        x=torch.tensor([[1],[4],[2],[9],[3]]);y=torch.ones_like(x)
        batches=[(x[:3],y[:3]),(x[3:],y[3:])]
        budgets=[1,16,17,32]
        combined={limit:{} for limit in budgets}
        for result in iter_inference(model,batches,2,budgets,16):
            for j,i in enumerate(result.indices.tolist()):
                self.assertNotIn(i,combined[result.cycle_limit])
                combined[result.cycle_limit][i]=(result.predictions[j].tolist(),int(result.cycles[j]))
        for limit in budgets:
            independent={}
            for result in single_budget(model,batches,2,limit,16):
                for j,i in enumerate(result.indices.tolist()): independent[i]=(result.predictions[j].tolist(),int(result.cycles[j]))
            self.assertEqual(combined[limit],independent)
            self.assertEqual(set(independent),set(range(5)))
        self.assertEqual(combined[32][0][1],17)  # 16 unchanged comparisons, not 16 predictions
        self.assertGreater(combined[32][3][1],17)

    def test_local_tiny_checkpoint_all_budgets_and_no_overwrite(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';data.mkdir();checkpoint=root/'model.pt';output=root/'records'
            arch=dict(name='hrm.models.hrm@HRM',num_layers=1,hidden_size=8,intermediate_size=16,head_dim=4,norm_eps=1e-6,rope_theta=10000.,H_cycles=2,L_cycles=6,bptt=True,forward_dtype='float32',is_causal=False,vocab_size=10,seq_len=82)
            torch.save(HRM(arch).state_dict(),checkpoint)
            config=dict(arch=arch,data=dict(name='hrm.tasks.sudoku',dataset_name='/missing/old/path',eval_split='test_hard'),cycles_per_data=16,local_batch_size=2,seeds=[1794189000])
            (root/'model_config.json').write_text(json.dumps(config))
            (data/'test_hard.csv').write_text('question,answer\n'+('0'*81+','+'123456789'*9+'\n')*3)
            argv=['--checkpoint',str(checkpoint),'--dataset-path',str(data),'--output-dir',str(output),'--device','cpu','--max-cycles','1','3','20','--stable-cycles','16','--save-predictions']
            evaluate(parse_args(argv))
            plan=json.loads((output/'run_config.json').read_text());self.assertEqual(plan['samples'],3);self.assertEqual(plan['evaluation_seed'],1794189000)
            combined=json.loads((output/'summary.json').read_text());self.assertEqual(set(combined['results']),{'1','3','20'})
            for limit in [1,3,20]:
                with np.load(output/f'cycles_{limit}/rank_00000.npz') as z:shard={k:z[k] for k in ['indices','correctness','cycles']}
                measured=validate_complete_results([shard],3,limit,16)
                self.assertEqual(measured['correct'],combined['results'][str(limit)]['correct'])
            with self.assertRaisesRegex(RuntimeError,'frozen evaluation'):evaluate(parse_args(argv))
            smaller=root/'limited'
            evaluate(parse_args(argv+['--limit','2','--data-seed','111','--output-dir',str(smaller)]))
            plan=json.loads((smaller/'run_config.json').read_text())
            self.assertEqual(plan['samples'],2)
            self.assertEqual(plan['resolved_data_options']['subsample_seed'],111)
            self.assertEqual(plan['resolved_data_options']['num_workers'],0)

    def test_incomplete_or_duplicate_shards_rejected(self):
        good=dict(indices=np.arange(3),correctness=np.array([True,False,True]),cycles=np.array([17,20,17]))
        self.assertEqual(validate_complete_results([good],3,20,16)['correct'],2)
        bad=dict(good,indices=np.array([0,0,2]))
        with self.assertRaises(ValueError):validate_complete_results([bad],3,20,16)
        with self.assertRaises(ValueError):validate_complete_results([good],3,20,0)

if __name__=='__main__':unittest.main()
