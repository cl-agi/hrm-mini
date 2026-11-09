import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import torch
from hrm.paths import repository_root
from hrm.training import data_scaling as trainer


class DataScalingTrainingTests(unittest.TestCase):
    def test_configs_preserve_fixed_budget_and_original_selection(self):
        for family in ('hrm','rt'):
            cfg=json.loads((repository_root()/f'exp/sudoku/configs/stable_stopping/{family}.json').read_text())
            cfg.pop('training_policy');cfg.update(seeds=[1794189600],output_dir='unused')
            c=trainer.TrainConfig(**cfg)
            self.assertEqual(c.total_steps//c.num_evals,4160)
            self.assertEqual(c.local_batch_size*8,768)
            self.assertEqual(c.data.subsample_seed,42)
            self.assertEqual(c.eval_cycles,16)
            self.assertEqual(c.ema,.999)
            with self.assertRaises(ValueError):trainer.TrainConfig(**dict(cfg,total_steps=83201))

    def test_public_evaluation_keeps_full_upstream_split(self):
        from hrm.evaluation.config import load_config
        with tempfile.TemporaryDirectory() as d:
            cfg=json.loads((repository_root()/'exp/sudoku/configs/stable_stopping/hrm.json').read_text())
            cfg['data']['prepared_data_path']='/missing/archive'
            path=Path(d)/'model_config.json';path.write_text(json.dumps(cfg))
            result=load_config(Path(d)/'best.pt',path,streaming=True,dataset='sapientinc/sudoku-extreme')
            self.assertEqual(result.data.dataset_name,'sapientinc/sudoku-extreme')
            self.assertNotIn('prepared_data_path',result.data.__pydantic_extra__)
            self.assertEqual(result.data.test_size,20000)

    def test_actual_cpu_updates_and_latest_tied_checkpoint(self):
        # Run the recovered controller with real tiny HRM/optimizer updates;
        # substitute CPU/DDP transport and controlled equal-score evaluations.
        class LocalDDP(torch.nn.Module):
            def __init__(self,model,**kw):super().__init__();self.module=model
            def forward(self,*a,**kw):return self.module(*a,**kw)
        cfg=json.loads((repository_root()/'exp/sudoku/configs/stable_stopping/hrm.json').read_text());cfg.pop('training_policy')
        cfg.update(seeds=[1794189600],total_steps=4,num_evals=2,cycles_per_data=2,eval_cycles=2,
                   local_batch_size=1,compile_model=False,lr_warmup_steps=0,log_interval=1)
        cfg['transformer'].update(hidden_size=16,intermediate_size=32,head_dim=8,forward_dtype='float32')
        cfg['arch'].update(num_layers=1,H_cycles=1,L_cycles=1)
        x=torch.tensor([[0,1,2]]);y=torch.tensor([[0,2,1]])
        class Loader:
            dataset=[dict(question='012',answer='021')]
            def __init__(self,split):self.split=split
            def __iter__(self):return itertools.repeat((x,y)) if self.split=='train' else iter([(x,y)])
        def loader(split,*a,**kw):
            return Loader(split),dict(vocab_size=10,seq_len=3,is_causal=False)
        real_load=trainer.load_module
        def load(name):return loader if 'create_dataloader' in name else real_load(name)
        class CPUTransport:
            # Override the controller's transport without replacing torch.device
            # globally: Dynamo requires that symbol to remain a type.
            def __getattr__(self,name):return getattr(torch,name)
            def device(self,*a,**kw):return torch.device('cpu')
            def zeros(self,*a,**kw):kw['device']='cpu';return torch.zeros(*a,**kw)
        def evaluate(*a,**kw):yield SimpleNamespace(predictions=y,targets=y,cycles=torch.tensor([2]))
        with tempfile.TemporaryDirectory() as d:
            cfg['output_dir']=d;c=trainer.TrainConfig(**cfg)
            with patch.object(trainer,'load_module',side_effect=load),patch.object(trainer,'DDP',LocalDDP),patch.object(trainer,'iter_inference',side_effect=evaluate),patch.object(trainer,'wandb',MagicMock()),patch.object(trainer.dist,'reduce'),patch.object(trainer,'torch',CPUTransport()),patch.object(torch.Tensor,'cuda',lambda self,*a,**kw:self):
                trainer.train_single_seed(c,c.seeds[0],'check',1,0)
            run=Path(d)/'check'/f'seed_{c.seeds[0]}'
            summary=json.loads((run/'training_summary.json').read_text())
            self.assertEqual(summary['optimizer_steps'],4)
            self.assertEqual(summary['trained_examples'],2)
            self.assertEqual(summary['best_eval']['optimizer_steps'],4)
            rows=[json.loads(line) for line in (run/'evaluation.jsonl').read_text().splitlines()]
            self.assertEqual([r['optimizer_step'] for r in rows],[2,4])
            first=torch.load(run/'step_2_acc_1.000000.pt',weights_only=True)
            last=torch.load(run/'step_4_acc_1.000000.pt',weights_only=True)
            self.assertTrue(any(not torch.equal(first[k],last[k]) for k in first))
            self.assertEqual(summary['best_eval']['eval']['test']['total'],1)

if __name__=='__main__':unittest.main()
