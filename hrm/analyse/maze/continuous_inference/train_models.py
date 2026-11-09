"""Matched fresh training; 16 optimizer-updated outer transitions per minibatch."""
import argparse,contextlib,datetime,json,math,os,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from hrm.analyse.maze.continuous_inference.data import HERE,SEED,batch,load_arrays,make_model,model_config,sat_metrics,digest
from hrm.analyse.maze.continuous_inference.metrics import metrics as maze_metrics
from hrm.training.optimizer import AdamATan2
from hrm.training.steps import forward


def write_json(path,value):
    p=Path(path);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(p)

def measure(task,p,b):
    return (sat_metrics if task=='sat' else maze_metrics)(p,b)

@torch.no_grad()
def evaluate(model,task,arrays,rank,world,bs=16):
    model.eval();tot=torch.zeros(4,device='cuda',dtype=torch.float64)
    ids=np.arange(len(arrays['ids']))[rank::world]
    for start in range(0,len(ids),bs):
        # Same fixed, independently seeded augmentation for every model/epoch.
        # Each puzzle's representation is generated separately, independent of batch size.
        rows=[batch(task,arrays,np.array([idx]),SEED*100000+int(arrays['ids'][idx]),device='cuda') for idx in ids[start:start+bs]]
        b={k:torch.cat([r[k] for r in rows]) if isinstance(rows[0][k],torch.Tensor) else rows[0][k] for k in rows[0]}
        c=model.initial_carry
        for _ in range(16):c,z=forward(model,c,b['x'],attn_mask=b['mask'])
        v,e=measure(task,z.argmax(-1),b)
        lim=513 if task=='sat' else 901
        loss=F.cross_entropy(z[:,:lim].flatten(0,1).float(),b['y'][:,:lim].flatten(),reduction='none').view(len(rows),lim).mean(1)
        tot+=torch.stack([v.sum(),e.sum(),loss.sum(),torch.tensor(len(rows),device='cuda')]).to(tot.dtype)
    if world>1:dist.all_reduce(tot)
    v,e,loss,n=tot.tolist();model.train()
    return dict(valid=v/n,exact=e/n,ce=loss/n,n=int(n),valid_count=int(v))

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['maze','sat'],required=True);p.add_argument('--kind',choices=['hrm','trm','rt'],required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--local-batch',type=int);p.add_argument('--repeat',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--out');p.add_argument('--dataset-path',type=Path);p.add_argument('--data-dir',type=Path);p.add_argument('--seed',type=int);a=p.parse_args()
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    from hrm.paths import repository_root
    from hrm.training.data import _freeze_seed
    output=Path(a.out) if a.out else repository_root()/'exp/maze/records/continuous/models'/f'{a.task}_{a.kind}'
    training_seed=_freeze_seed(output,'training_seed.json',a.seed)
    if (output/'model_config.json').exists() and not a.resume:
        raise FileExistsError('Existing training run; use --resume or a new --out')
    torch.cuda.set_device(local);torch.set_num_threads(2);torch.manual_seed(training_seed);torch.cuda.manual_seed_all(training_seed)
    if world>1:dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))
    bs=a.local_batch or (96 if a.task=='maze' else 48);repeat=a.repeat or (25 if a.task=='maze' else 20)
    data_root=a.data_dir
    if a.task=='maze':
        from .prepare import prepare
        # All ranks use one prepared cohort, with preparation performed by rank zero.
        if rank==0:data_root=prepare(data_root,a.dataset_path)
        if world>1:dist.barrier()
        if data_root is None:data_root=repository_root()/'exp/maze/records/continuous/data'
    elif a.dataset_path is not None:
        raise ValueError('--dataset-path is supported for Maze preparation only; use --data-dir for prepared SAT')
    if data_root is None:data_root=HERE/'data'
    train=load_arrays(a.task,'train',data_root);val=load_arrays(a.task,'val',data_root);steps_epoch=(len(train['ids'])*repeat//(bs*world))*16
    assert steps_epoch>0
    output.mkdir(parents=True,exist_ok=True)
    config=dict(task=a.task,kind=a.kind,arch=model_config(a.task,a.kind),cycles_per_data=16,epochs=a.epochs,local_batch_size=bs,world_size=world,global_batch=bs*world,repeat=repeat,seed=training_seed,
        lr=1e-4,lr_warmup_steps=2000,lr_min_ratio=1.,beta1=.9,beta2=.95,weight_decay=1.,ema=.999,
        train_count=len(train['ids']),validation_count=len(val['ids']),steps_epoch=steps_epoch,total_steps=steps_epoch*a.epochs,
        selection='max validation semantic valid, tie lower validation CE, at exactly 16 outer cycles',fresh_initialization=True,
        data_manifest_sha256=digest(data_root/'manifest.json'),torch=torch.__version__)
    model=make_model(a.task,a.kind)
    optimizer=AdamATan2(model.parameters(),lr=torch.tensor(0.),betas=(.9,.95),weight_decay=1.,ema=.999)
    step=0;start_epoch=0;best=(-1.,float('-inf'));history=[]
    if a.resume and (output/'latest_train.pt').exists():
        state=torch.load(output/'latest_train.pt',map_location='cpu',weights_only=False)
        assert state['config']==config
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);step=state['step'];start_epoch=state['epoch']+1;best=tuple(state['best']);history=state['history']
    if rank==0:
        write_json(output/'model_config.json',config)
        if step==0:torch.save(model.state_dict(),output/'initial.pt')
    wrapped=DDP(model,device_ids=[local],static_graph=True,broadcast_buffers=False) if world>1 else model
    started=time.time()
    if rank==0:print(json.dumps({'event':'start','config':config,'params':sum(p.numel() for p in model.parameters())}),flush=True)
    for epoch in range(start_epoch,a.epochs):
        order=np.random.default_rng(training_seed+epoch).permutation(np.tile(np.arange(len(train['ids'])),repeat))
        order=order[:(len(order)//(bs*world))*(bs*world)].reshape(-1,world,bs)
        model.train();epoch_t=time.time()
        for bi,global_ids in enumerate(order):
            b=batch(a.task,train,global_ids[rank],training_seed+1000000*epoch+bi*world+rank)
            carry=model.initial_carry
            for outer in range(16):
                step+=1;lr=1e-4*min(1.,step/2000)
                for group in optimizer.param_groups:group['lr'].fill_(lr)
                carry,z=forward(wrapped,carry,b['x'],attn_mask=b['mask'])
                limit=513 if a.task=='sat' else 901
                loss=F.cross_entropy(z[:,:limit].flatten(0,1).float(),b['y'][:,:limit].flatten())
                loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
            loss_value=float(loss.detach())
            if not math.isfinite(loss_value):raise RuntimeError('Non-finite training loss')
            if rank==0 and (bi%4==0 or bi+1==len(order)):
                rec=dict(event='train',task=a.task,kind=a.kind,epoch=epoch,batch=bi+1,batches=len(order),step=step,total_steps=config['total_steps'],loss=loss_value,elapsed=time.time()-started,epoch_elapsed=time.time()-epoch_t,memory_gb=torch.cuda.max_memory_allocated()/1e9)
                write_json(output/'progress.json',rec);print(json.dumps(rec),flush=True)
        optimizer.swap_ema()
        vm=evaluate(model,a.task,val,rank,world)
        chosen=(vm['valid'],-vm['ce'])>best
        if chosen:best=(vm['valid'],-vm['ce'])
        if rank==0:
            torch.save(model.state_dict(),output/f'epoch_{epoch}.pt')
            if chosen:
                torch.save(model.state_dict(),output/'best.pt');write_json(output/'best.json',dict(epoch=epoch,step=step,validation=vm,checkpoint_sha256=digest(output/'best.pt')))
            rec=dict(event='epoch',epoch=epoch,step=step,validation=vm,selected=chosen,seconds=time.time()-epoch_t,elapsed=time.time()-started)
            history.append(rec);write_json(output/'history.json',history);print(json.dumps(rec),flush=True)
        optimizer.swap_ema()
        if rank==0:
            tmp=output/'latest_train.pt.tmp'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,step=step,best=best,history=history,config=config),tmp);tmp.replace(output/'latest_train.pt')
        if world>1:dist.barrier()
    if rank==0:write_json(output/'complete.json',dict(status='complete',step=step,elapsed=time.time()-started,best=json.loads((output/'best.json').read_text())))
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
