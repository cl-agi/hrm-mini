import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from hrm.models.hrm import HRM
from hrm.analyse.sudoku.continuous_inference.runner_snapshot import parse_args,main,summarize,sudoku_valid


class ContinuousSudokuTests(unittest.TestCase):
    def test_reversals_count_after_solution_not_just_final_endpoint(self):
        trace=np.zeros((2,4,3),dtype=bool)
        trace[0,:,0]=[False,True,False,True];trace[0,:,2]=trace[0,:,0]
        trace[1,:,0]=[False,False,True,True];trace[1,:,2]=trace[1,:,0]
        rows=summarize(trace,np.array([2,3]),[2,4])
        self.assertEqual(rows[0]['exact_reversal_examples'],0)
        self.assertEqual(rows[1]['exact_reversal_examples'],1)
        self.assertEqual(rows[1]['valid_reversal_examples'],1)
        self.assertEqual(rows[1]['exact_all82'],1.)

    def test_config_architecture_variants_load_strictly(self):
        from hrm.analyse.sudoku.continuous_inference.runner_snapshot import load_metadata
        from hrm.registry import model_class
        for module,cls,extra in [('hrm.models.rt','RecurrentTransformer',dict(cycles=2)),
                                 ('hrm.models.ablations','HRM',dict(H_cycles=1,L_cycles=1,dual_module=False,hh_link=True,sym_io=False))]:
            with self.subTest(module=module),tempfile.TemporaryDirectory() as d:
                root=Path(d)
                backbone=dict(num_layers=1,hidden_size=8,intermediate_size=16,head_dim=4,forward_dtype='float32',norm_eps=1e-6,rope_theta=10000.,seq_len=82,vocab_size=10,is_causal=False)
                arch=dict(name=module+'@'+cls,bptt=True,**extra)
                model=model_class(arch['name'],arch)(backbone|arch)
                torch.save(model.state_dict(),root/'model.pt')
                (root/'model_config.json').write_text(json.dumps(dict(transformer=backbone,arch=arch)))
                args=parse_args([str(root/'model.pt'),'--device','cpu'])
                loaded,meta=load_metadata(args)
                self.assertEqual(meta['arch']['hidden_size'],8)
                for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,loaded.state_dict()[key]))

    def test_cpu_batch_records_do_not_alias_runner_buffers(self):
        from hrm.analyse.sudoku.continuous_inference.runner_snapshot import GraphRunner
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);arch=dict(name='hrm.models.hrm@HRM',num_layers=1,hidden_size=8,intermediate_size=16,head_dim=4,H_cycles=1,L_cycles=1,bptt=True,forward_dtype='float32',norm_eps=1e-6,rope_theta=10000.,seq_len=82,vocab_size=10)
            model=HRM(arch);torch.save(model.state_dict(),root/'model.pt');(root/'model_config.json').write_text(json.dumps(dict(arch=arch)))
            (root/'test.csv').write_text('question,answer\n'+('0'*81+','+'123456789'*9+'\n')*3)
            original=GraphRunner.transition;calls=[0]
            def marked_transition(runner):
                out=original(runner);calls[0]+=1
                # Distinct batch snapshots; test ownership independently of accuracy.
                marker=calls[0]%2==0
                runner.trace[runner.step[0]-1].fill_(marker)
                runner.first_valid.fill_(calls[0]);runner.first_valid_prediction.fill_(calls[0])
                return out
            # The production parity check is covered separately; injected markers
            # intentionally replace validity here to expose storage aliasing.
            with patch('hrm.analyse.sudoku.continuous_inference.runner_snapshot.parity_check',return_value={}),patch.object(GraphRunner,'transition',marked_transition):
                main(parse_args([str(root/'model.pt'),'--dataset',str(root/'test.csv'),'--device','cpu','--max-examples','3','--batch-size','1','--max-cycles','1','--seed','1794201000','--output',str(root/'result.json')]))
            with np.load(root/'result.npz') as merged:
                for i in range(3):
                    with np.load(root/f'result.batch_{i}.npz') as batch:
                        for key in ('exact_all82','exact_board81','valid','first_valid_cycle','first_valid_prediction'):
                            np.testing.assert_array_equal(merged[key][i:i+1],batch[key])

    def test_cpu_native_carry_and_prefix_repeatability(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);arch=dict(name='hrm.models.hrm@HRM',num_layers=1,hidden_size=8,intermediate_size=16,head_dim=4,H_cycles=1,L_cycles=1,bptt=True,forward_dtype='float32',norm_eps=1e-6,rope_theta=10000.,seq_len=82,vocab_size=10,is_causal=False)
            torch.manual_seed(123);torch.save(HRM(arch).state_dict(),root/'model.pt')
            (root/'model_config.json').write_text(json.dumps(dict(arch=arch,cycles_per_data=16)))
            (root/'test.csv').write_text('question,answer\n'+('0'*81+','+'123456789'*9+'\n')*3)
            base=[str(root/'model.pt'),'--dataset',str(root/'test.csv'),'--seed','123','--device','cpu','--batch-size','2','--max-examples','3']
            main(parse_args(base+['--max-cycles','2','--output',str(root/'short.json')]))
            main(parse_args(base+['--max-cycles','4','--output',str(root/'long.json'),'--prefix-reference',str(root/'short.npz')]))
            result=json.loads((root/'long.json').read_text())
            self.assertTrue(result['parity']['logits_bitwise_equal'])
            self.assertFalse(result['runtime']['cuda_graph'])
            self.assertEqual(len(result['prefix_audits']),2)
            with np.load(root/'short.npz') as a,np.load(root/'long.npz') as b:
                self.assertEqual(b['exact_all82'].shape,(3,4))
                np.testing.assert_array_equal(a['exact_all82'],b['exact_all82'][:,:2])
                self.assertEqual(len(set(b['indices'])),3)
            with self.assertRaises(FileExistsError):main(parse_args(base+['--max-cycles','4','--output',str(root/'long.json')]))

if __name__=='__main__':unittest.main()
