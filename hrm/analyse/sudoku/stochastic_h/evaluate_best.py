import os,json,time,argparse
from pathlib import Path
import yaml,torch
import torch.distributed as dist
from hrm.training.trainer import load_module
from hrm.analyse.sudoku.stochastic_h.sudoku_validity import valid_sudoku
from hrm.analyse.sudoku.stochastic_h.early_valid_pool import evaluate_pool

def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint_dir',type=Path);a=p.parse_args()
 local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
 dist.init_process_group('nccl',device_id=torch.device('cuda',local));rank,world=dist.get_rank(),dist.get_world_size();assert world==8
 cfg=yaml.safe_load((a.checkpoint_dir/'model_config.json').read_text());summary=json.loads((a.checkpoint_dir/'training_summary.json').read_text());assert summary['complete']
 ck=a.checkpoint_dir/summary['best']['checkpoint'];data=dict(cfg['data']);name=data.pop('name')
 loader,metadata=load_module(f'dataset.{name}@create_dataloader')('test_hard',96,rank=rank,world_size=world,seed=20261017,**data)
 arch=dict(cfg['arch']);name=arch.pop('name')
 with torch.device('cuda'):native=load_module(f'arch.{name}')(arch|metadata)
 native.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=True)
 model=torch.compile(native,dynamic=True,fullgraph=True);model.eval();torch.manual_seed(20261017+rank)
 validate=torch.compile(valid_sudoku,dynamic=True,fullgraph=True);inputs=torch.cat([x for x,y in loader]).cuda();assert len(inputs)==2500
 out=Path('evaluation');out.mkdir(exist_ok=True);start=time.monotonic()
 def progress(iteration,completed,admitted,active,hits,used):
  r=dict(rank=rank,iteration=iteration,completed=completed,admitted=admitted,active=active,solved=int((hits>0).sum()),seconds=time.monotonic()-start)
  (out/f'rank{rank}_progress.json').write_text(json.dumps(r));print(json.dumps(r),flush=True)
 hits,used=evaluate_pool(model,inputs,validate,batch_size=96,max_outer=4096,progress=progress)
 torch.save(dict(first_valid_outer=hits.cpu(),outers_used=used.cpu()),out/f'rank{rank}.pt')
 depths=[16,32,64,128,256,512,1024,2048,4096];counts=torch.stack([((hits>0)&(hits<=d)).sum() for d in depths]);totals=torch.tensor([len(inputs),int(used.sum())],device='cuda');dist.all_reduce(counts);dist.all_reduce(totals)
 if rank==0:
  num,updates=totals.cpu().tolist();assert num==20000
  r=dict(complete=True,checkpoint=str(ck),selection_depth=128,total=num,world_size=world,seed=20261017,mean_outers=updates/num,total_row_updates=updates,seconds=time.monotonic()-start,criterion='first valid Sudoku preserving givens; one trajectory; early exit',eval={str(d):dict(any_valid=int(v),any_valid_rate=int(v)/num) for d,v in zip(depths,counts.cpu().tolist())})
  (out/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
 dist.destroy_process_group()
if __name__=='__main__':main()
