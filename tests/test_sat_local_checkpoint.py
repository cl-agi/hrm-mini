import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from hrm import hub
from hrm.tasks.sat_flow import SATTask
from hrm.analyse.sat.evaluate import load_checkpoint,evaluate


class SATCheckpointTests(unittest.TestCase):
    def config(self):
        config=copy.deepcopy(hub.sat_training_config())
        config['arch'].update(hidden_size=32,intermediate_size=64,head_dim=8,num_layers=1,
                              H_cycles=1,L_cycles=1,seq_len=8,forward_dtype='float32')
        config['seed']=1794178200
        return config

    def test_saved_config_and_explicit_raw_ema_roundtrip(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'weights').mkdir();cfg=self.config()
            (root/'resolved_config.json').write_text(json.dumps(cfg))
            task=SATTask(cfg,root,device='cpu');model=task.model()
            arrays=dict(clauses=np.array([[[1,2,3]]]),answers=np.array([[1,1,1]]))
            batch=task.batch(arrays,[0],cfg['seed'],True);task.bind(model,batch['mask'])
            optimizer=torch.optim.SGD(model.parameters(),lr=.01)
            carry,z=model(model.initial_carry|{'mixture_weights':batch['mix']},batch['x'])
            task.ce_cases(z,batch).mean().backward();optimizer.step()
            ema={k:v.detach().clone()+.125 for k,v in model.named_parameters()}
            path=root/'weights/s1.pt';torch.save(dict(model=model.state_dict(),ema=ema),path)
            raw,actual,source=load_checkpoint(path)
            self.assertEqual(actual,cfg);self.assertTrue(source.endswith('resolved_config.json'))
            averaged,_,_=load_checkpoint(path,weight_state='ema')
            for key,value in raw.named_parameters():
                self.assertTrue(torch.equal(value,dict(model.named_parameters())[key]))
                self.assertTrue(torch.equal(dict(averaged.named_parameters())[key],ema[key]))
            for density in ('4.2','4.3'):
                np.savez(root/f'data_{density}.npz',clauses=arrays['clauses'],answers=np.zeros((1,3),dtype=int),official_ids=np.array([7]))
            protocol=hub.sat_eval_config(1024);protocol['microbatch']=1
            for state in ('raw','ema'):
                result=evaluate(protocol,None,root,checkpoint=path,device='cpu',cycles=1,starts=1,weight_state=state)
                self.assertEqual(result['weight_state'],state)
                self.assertEqual(result['training_seed'],cfg['seed'])
                self.assertTrue(all(x['total']==1 for x in result['subsets'].values()))
            plain=root/'plain.pt';torch.save(model.state_dict(),plain)
            with self.assertRaisesRegex(ValueError,'no EMA'):load_checkpoint(plain,cfg,weight_state='ema')
            embedded=root/'latest.pt';torch.save(dict(model=model.state_dict(),ema=ema,config=cfg),embedded)
            self.assertEqual(load_checkpoint(embedded)[2],'checkpoint.config')


if __name__=='__main__':unittest.main()
