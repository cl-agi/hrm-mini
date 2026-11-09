from pathlib import Path
import json,numpy as np,torch
from hrm.training.optimizer import AdamATan2
from hrm.tasks.sat_flow import SATTask,sat_batch,sat_satisfies
from hrm.training import sat_validation

def write(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def train(config, output, data_root):
    global P,CFG,model,bind,arrays,batch,ce_cases,validate
    P=Path(output);P.mkdir(parents=True,exist_ok=True);CFG=config
    task=SATTask(config,data_root)
    model,bind,arrays,batch,ce_cases=(getattr(task,k) for k in ['model','bind','arrays','batch','ce_cases'])
    for k,v in dict(P=P,CFG=CFG,arrays=arrays,batch=batch,bind=bind,sat_satisfies=sat_satisfies).items():setattr(sat_validation,k,v)
    validate=sat_validation.validate
    main()


import os,time,datetime,hashlib
import torch.distributed as dist

def log(x):
 if dist.get_rank()==0:
  with (P/'events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
  print(json.dumps(x),flush=True)
def main():
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);assert world==8;torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2);dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=40));torch._dynamo.config.cache_size_limit=64
 m=model();opt=AdamATan2(m.parameters(),lr=torch.tensor(0.),betas=(CFG['beta1'],CFG['beta2']),weight_decay=CFG['weight_decay'],ema=.999)
 ema={k:v.detach().clone() for k,v in m.named_parameters()};step=0;n=CFG['n_vars'];micro=CFG['microbatch'];a=arrays('train',n);streams=[];ck=None
 if (P/'latest.pt').exists():
  ck=torch.load(P/'latest.pt',map_location='cpu',weights_only=False);assert {k:v for k,v in ck['config'].items() if k!='validation_every'}=={k:v for k,v in CFG.items() if k!='validation_every'};m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step=ck['step'];ema={k:v.cuda() for k,v in ck['ema'].items()}
  for g in opt.param_groups:
   g['lr']=g['lr'].cpu()
   for x in g['params']:
    if 'step' in opt.state[x]:opt.state[x]['step']=opt.state[x]['step'].cpu()
  torch.set_rng_state(ck['rng'][rank]['cpu']);torch.cuda.set_rng_state(ck['rng'][rank]['cuda'])
 def fresh(count,seed):
  ids=np.random.default_rng(seed).integers(0,len(a['answers']),count);return batch(a,ids,seed,True)
 if ck is not None:
  streams=torch.load(P/ck['stream_dir']/f'rank{rank}.pt',map_location='cuda',weights_only=False)
  for st in streams:
   b=st['b'];b['mask']=sat_batch(b['clauses'],b['y'][:,1:2*n+1:2]-1,torch.Generator(device='cuda').manual_seed(0),False,0.)['mask']
 else:
  for j in range(CFG['local_batch_size']//micro):
   b=fresh(micro,80000000+rank*100+j);streams.append(dict(b=b,carry=m.initial_carry|{'mixture_weights':b['mix']},age=torch.zeros(micro,device='cuda',dtype=torch.long),streak=torch.zeros(micro,device='cuda',dtype=torch.long)))
 del ck
 run=torch.compile(m,dynamic=False,fullgraph=True);started=time.time();deadline=time.time()+CFG.get('wall_time_seconds',float('inf'));stats=torch.zeros(7,device='cuda',dtype=torch.float64)
 def save():
  dirname=f'stream_s{step}';out=P/dirname;out.mkdir(exist_ok=True);states=[dict(b={k:v for k,v in s['b'].items() if k!='mask'},carry=s['carry'],age=s['age'],streak=s['streak']) for s in streams];tmp=out/f'rank{rank}.tmp';torch.save(states,tmp);tmp.replace(out/f'rank{rank}.pt')
  rngs=[None]*world;dist.all_gather_object(rngs,dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()))
  if rank==0:
   tmp=P/'latest.pt.tmp';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),ema=ema,step=step,stream_dir=dirname,config=CFG,rng=rngs),tmp);tmp.replace(P/'latest.pt');write(P/'progress.json',dict(step=step,cap=CFG['cycles_per_data'],n=n,elapsed=time.time()-started))
  dist.barrier()
 log(dict(event='start',step=step,n=n,cap=CFG['cycles_per_data'],head=False,halt='3 consecutive valid assignments'));save()
 while True:
  stop=torch.tensor(int(step>=CFG['target_updates'] or time.time()>=deadline or (P/'PAUSE').exists()) if rank==0 else 0,device='cuda');dist.broadcast(stop,0)
  if stop.item():break
  step+=1;t=time.time();pending=[]
  for st in streams:
   s=st['b'];bind(m,s['mask']);st['carry'],z=run(st['carry'],s['x']);ce=ce_cases(z,s).mean();assert torch.isfinite(ce);(ce/len(streams)).backward()
   with torch.no_grad():
    pred=z[:,1:2*n+1:2].argmax(-1);ok=sat_satisfies(s['clauses'],pred==2)&((pred==1)|(pred==2)).all(-1);st['age']+=1;st['streak']=torch.where(ok,st['streak']+1,0);halt=st['streak']>=3;timeout=st['age']>=CFG['cycles_per_data'];replace=halt|timeout;pending.append(replace)
    stats+=torch.stack([ce.detach().double()*micro,ok.sum().double(),halt.sum().double(),(timeout&~halt).sum().double(),replace.sum().double(),(st['age']*replace).sum().double(),torch.tensor(float(micro),device='cuda',dtype=torch.float64)])
  for x in m.parameters():assert x.grad is not None and torch.isfinite(x.grad).all();dist.all_reduce(x.grad);x.grad.div_(world)
  lr=CFG['lr']*min(1.,step/CFG['lr_warmup_steps']);opt.param_groups[0]['lr'].fill_(lr);opt.step();opt.zero_grad(set_to_none=True)
  with torch.no_grad():
   for k,x in m.named_parameters():ema[k].lerp_(x,1-CFG['ema'])
   for j,(st,replace) in enumerate(zip(streams,pending)):
    count=int(replace.sum())
    if count:
     nb=fresh(count,900000000+step*world*len(streams)+rank*len(streams)+j)
     for k,v in st['b'].items():
      if isinstance(v,torch.Tensor):v[replace]=nb[k]
     for k in list(m.initial_carry):st['carry'][k]=torch.where(replace[:,None,None],m.initial_carry[k],st['carry'][k])
     st['carry']['mixture_weights']=st['b']['mix'];st['age'][replace]=0;st['streak'][replace]=0
  if step%16==0 or step==1:
   v=stats.clone();dist.all_reduce(v);log(dict(event='train',step=step,lr=lr,ce=float(v[0]/v[6]),train_valid=float(v[1]/v[6]),three_valid_halts=int(v[2]),timeouts=int(v[3]),replacements=int(v[4]),mean_completed_age=float(v[5]/v[4]) if v[4]>0 else None,last_update_seconds=time.time()-t));stats.zero_()
  if step%256==0 or step==CFG['startup_validation_step']:save()
  if step%CFG['validation_every']==0 or step==CFG['startup_validation_step'] or step==CFG['target_updates']:
   save()
   startup=step==CFG['startup_validation_step']; eval_starts=CFG['startup_validation_starts'] if startup else CFG['validation_starts']; prefix='startup_' if startup else ''
   for cap in [16,64]:
    sc=validate(m,run,rank,world,step,prefix+f'raw_outer{cap}',cap,eval_starts);log(dict(event='startup_validation' if startup else 'validation',step=step,kind=prefix+f'raw_outer{cap}',score=sc))
   original={k:x.detach().clone() for k,x in m.named_parameters()}
   with torch.no_grad():
    for k,x in m.named_parameters():x.copy_(ema[k])
   sc=validate(m,run,rank,world,step,prefix+'ema_outer64',64,eval_starts);log(dict(event='startup_validation' if startup else 'validation',step=step,kind=prefix+'ema_outer64',score=sc))
   with torch.no_grad():
    for k,x in m.named_parameters():x.copy_(original[k])
   del original
   if rank==0:
    (P/'weights').mkdir(exist_ok=True);torch.save(dict(model=m.state_dict(),ema=ema),P/'weights'/f's{step}.pt')
   save()
 save()
 if rank==0:write(P/'complete.json',dict(step=step,paused=step<CFG['target_updates'],reason='target_updates' if step>=CFG['target_updates'] else ('deadline' if time.time()>=deadline else 'PAUSE')))
 dist.destroy_process_group()

