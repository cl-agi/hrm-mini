from pathlib import Path
import json,torch,numpy as np
import os,time,datetime,hashlib,math
import torch.distributed as dist

def log(r):
 if dist.get_rank()==0:
  with (P/'events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  print(json.dumps(r),flush=True)
@torch.no_grad()
def validate(m,run,rank,world,step,kind,eval_outer=None,eval_starts=None):
 eval_starts=CFG['validation_starts'] if eval_starts is None else eval_starts
 eval_outer=CFG['validation_outer'] if eval_outer is None else eval_outer
 cr=torch.get_rng_state();gr=torch.cuda.get_rng_state();m.eval();scores={}
 for n in CFG['validation_sizes']:
  a=arrays('valid',n);ids=np.arange(rank,len(a['clauses']),world);micro=8 if n<1000 else 2;preds=[]
  for start in range(eval_starts):
   pp=[]
   for lo in range(0,len(ids),micro):
    ii=ids[lo:lo+micro];pad=np.r_[ii,np.repeat(ii[0],micro-len(ii))];rows=[batch(a,[int(i)],801000+int(i)+100000*start,False) for i in pad];b={k:torch.cat([r[k] for r in rows]) for k in ['x','mix','mask']};bind(m,b['mask']);carry=m.initial_carry|{'mixture_weights':b['mix']}
    for it in range(eval_outer):carry,z=run(carry,b['x'])
    pp.append(z[:len(ii),1:2*n+1:2].argmax(-1).cpu().numpy())
   preds.append(np.concatenate(pp) if pp else np.empty((0,n),dtype=np.int64))
  pred=np.stack(preds);c=a['clauses'][ids].astype(int);truth=np.take_along_axis(pred==2,np.broadcast_to(abs(c)[None]-1,(len(pred),len(ids),a['clauses'].shape[1],3)).reshape(len(pred),len(ids),a['clauses'].shape[1]*3),axis=2).reshape(len(pred),len(ids),a['clauses'].shape[1],3)
  sat=np.any(truth==(c[None]>0),axis=-1);legal=sat.all(-1)&np.isin(pred,[1,2]).all(-1)
  gt=sat_satisfies(torch.tensor(c,device='cuda').repeat(len(pred),1,1),torch.tensor(pred==2,device='cuda').flatten(0,1)).view(legal.shape).cpu().numpy()&np.isin(pred,[1,2]).all(-1);assert np.array_equal(gt,legal)
  out=P/'validation_predictions'/kind/f'n{n}';out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/f's{step}_rank{rank}.npz',ids=ids,pred=pred,valid=legal)
  v=torch.tensor([legal.any(0).sum(),legal.sum(),len(ids),(~sat).sum()],device='cuda',dtype=torch.float64);dist.all_reduce(v);scores[str(n)]=dict(solved=int(v[0]),n=int(v[2]),rate=float(v[0]/v[2]),mean_single=float(v[1]/(v[2]*len(pred))),mean_violated=float(v[3]/(v[2]*len(pred))))
 torch.set_rng_state(cr);torch.cuda.set_rng_state(gr);m.train();return dict(by_size=scores,macro_rate=float(np.mean([x['rate'] for x in scores.values()])),starts=eval_starts,outer=eval_outer,mean_used_outer=eval_outer,independent_verifier_match=True)
