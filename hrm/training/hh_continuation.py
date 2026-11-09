"""E30 matched full/deleted-HH continuation from raw optimizer/EMA state."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from hrm.paths import repository_root
from hrm.models.ablations import HRM
from hrm.training.optimizer import AdamATan2
from hrm.training.steps import train_step, run_inference
from hrm.training.data import puzzle_directory, local_directory, _freeze_seed
from hrm.tasks.sudoku_stochastic_h import create_dataloader, collate_fn
from hrm.analyse.dynamics.jacobian.data import read_test_records, resolve_test_csv
from hrm.analyse.dynamics.jacobian.models import analysis_architecture
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha


def restore(model, optimizer, saved):
    """Restore raw weights and all Adam/EMA buffers; weights-only is not continuation."""
    if not {'model_state_dict','optimizer_state_dict','epoch','step'} <= saved.keys():
        raise ValueError('E30 requires a full training checkpoint with raw weights, optimizer, EMA, epoch and step; best/last weights alone are insufficient')
    states=saved['optimizer_state_dict']['state']
    if not states or any(not {'step','exp_avg','exp_avg_sq','param_ema'} <= s.keys() for s in states.values()):
        raise ValueError('Missing Adam momentum or EMA buffers')
    for group in saved['optimizer_state_dict']['param_groups']:
        if tuple(group['betas'])!=(.9,.95) or group['ema']!=.999 or group['weight_decay']!=1.:
            raise ValueError('Source optimizer differs from the E30 protocol')
    model.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in saved['model_state_dict'].items()},strict=True)
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    for parameter,state in optimizer.state.items():
        for key in ('exp_avg','exp_avg_sq','param_ema'):
            if state[key].shape!=parameter.shape or not torch.isfinite(state[key]).all():raise ValueError('Invalid optimizer buffer: '+key)
        state['step']=state['step'].cpu()
    for group in optimizer.param_groups:group['lr']=torch.tensor(1e-4,device='cpu')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path)
    p.add_argument('--branch',choices=['full','deleted'],required=True)
    p.add_argument('--dataset',type=Path);p.add_argument('--output-dir',type=Path,default=repository_root()/'exp/memory/records/old_h/continuation')
    p.add_argument('--seed',type=int,help='Use the same frozen seed for the paired full/deleted branches')
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--global-batch',type=int,default=768)
    p.add_argument('--evaluation-puzzles',type=int,default=512);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if min(a.epochs,a.global_batch,a.evaluation_puzzles)<1 or a.global_batch%world:raise ValueError('Positive sizes and global batch divisible by world size required')
    if (a.output_dir/'manifest.json').exists():raise FileExistsError(a.output_dir)
    source=a.source.expanduser().resolve();config_path=a.config or source.with_name('model_config.json')
    cfg=yaml.safe_load(config_path.read_text());arch=analysis_architecture(cfg)
    if not arch.get('dual_module',True) or arch.get('sym_io',False) or not arch.get('hh_link',True) or arch['H_cycles']!=2 or arch['L_cycles']!=6:raise ValueError('Source must be full-link 2H6L HRM')
    if cfg.get('cycles_per_data')!=16:raise ValueError('Source must use 16 training cycles per batch')
    source_hash=sha(source);saved=torch.load(source,map_location='cpu',weights_only=True)
    # Validate the complete state even for prepare-only; never substitute public EMA weights.
    arch['hh_link']=a.branch=='full';base=HRM(arch)
    optimizer=AdamATan2(base.parameters(),lr=torch.tensor(1e-4),betas=(.9,.95),weight_decay=1.,ema=.999)
    restore(base,optimizer,saved)
    dataset=local_directory(a.dataset) if a.dataset is not None else puzzle_directory('sudoku')
    if not (dataset/'train.csv').is_file() or not (dataset/'test_hard.csv').is_file():raise FileNotFoundError('Dataset requires train.csv and test_hard.csv')
    records=read_test_records(dataset)
    if a.evaluation_puzzles>len(records):raise ValueError('Not enough evaluation puzzles')
    seed=_freeze_seed(a.output_dir,'training_seed.json',a.seed)
    ids=np.random.default_rng(seed).permutation(len(records))[:a.evaluation_puzzles].tolist()
    data={k:v for k,v in cfg['data'].items() if k not in ('name','dataset_name')}
    if set(data)-{'augment','repeat','partial_fill_probability','partial_fill_remaining_blanks'} or data.get('partial_fill_probability',0)!=0:raise ValueError('Unsupported source data policy')
    plan=dict(source=str(source),source_sha256=source_hash,source_epoch=saved['epoch'],source_step=saved['step'],
        source_world=saved.get('world_size'),branch=a.branch,seed=seed,epochs=a.epochs,global_batch=a.global_batch,world_size=world,
        evaluation_ids=ids,dataset=str(dataset),train_sha256=sha(dataset/'train.csv'),test_sha256=sha(resolve_test_csv(dataset)),
        device=a.device,lr=1e-4,warmup=False,cycles_per_data=16,optimizer_and_ema_restored=True,
        data_policy=data,source_config=cfg,code_sha256=sha(Path(__file__)))
    plan_path=a.output_dir/'run_config.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Prepared continuation settings changed; use a new output directory')
    if rank==0:plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    if a.prepare_only:return 0
    if a.device=='cuda':torch.cuda.set_device(local)
    device=torch.device('cuda',local) if a.device=='cuda' else torch.device('cpu')
    if world>1:dist.init_process_group('nccl' if a.device=='cuda' else 'gloo')
    torch.set_num_threads(2);torch.manual_seed(seed)
    base=base.to(device)
    # Loading after placement moves momentum and EMA buffers with their parameters.
    optimizer=AdamATan2(base.parameters(),lr=torch.tensor(1e-4),betas=(.9,.95),weight_decay=1.,ema=.999);restore(base,optimizer,saved);del saved
    model=DDP(base,device_ids=[local] if a.device=='cuda' else None,static_graph=True) if world>1 else base
    loader,_=create_dataloader('train',a.global_batch//world,rank,world,dataset_name=str(dataset),seed=seed,**data)
    if len(loader)==0:raise ValueError('No complete training batch')
    if rank==0:
        model_config=dict(cfg,arch=dict(arch,name='hrm.models.ablations@HRM'),data=dict(data,name='hrm.tasks.sudoku_stochastic_h',dataset_name=str(dataset)),seeds=[seed],epochs=a.epochs,local_batch_size=a.global_batch//world)
        (a.output_dir/'model_config.json').write_text(json.dumps(model_config,indent=2))
        (a.output_dir/'manifest.json').write_text(json.dumps(dict(status='running',plan_sha256=sha(plan_path))))
    step=0
    def evaluate(epoch):
        model.eval();optimizer.swap_ema()
        count=torch.zeros(2,dtype=torch.long,device=device)
        with torch.inference_mode():
            # Call base (no DDP collectives): rank-strided eval may have unequal batches.
            chosen=ids[rank::world]
            for start in range(0,len(chosen),a.global_batch//world):
                rows=[vars(records[i]) for i in chosen[start:start+a.global_batch//world]]
                x,y=collate_fn(rows,False);x=x.to(device);y=y.to(device);carry=base.initial_carry
                for _ in range(16):carry,pred=run_inference(base,carry,x)
                count+=torch.stack(((pred==y).all(-1).sum(),torch.tensor(len(x),device=device)))
        if world>1:dist.all_reduce(count)
        if rank==0:
            if int(count[1])!=len(ids):raise ValueError('Evaluation coverage mismatch')
            torch.save(base.state_dict(),a.output_dir/f'epoch_{epoch}.pt')
            with (a.output_dir/'evaluation.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,continuation_step=step,seed=seed,correct=int(count[0]),total=int(count[1]),weights='EMA',cycles=16,split='test_hard',optimizer_step=step,inference_cycles=16,accuracy=int(count[0])/int(count[1])))+'\n')
        optimizer.swap_ema()
    evaluate(0)
    for epoch in range(1,a.epochs+1):
        model.train()
        for x,y in loader:
            x=x.to(device);y=y.to(device);carry=base.initial_carry
            for _ in range(16):carry,_=train_step(model,carry,optimizer,x,y);step+=1
        evaluate(epoch)
        if rank==0:
            torch.save(dict(model_state_dict=base.state_dict(),optimizer_state_dict=optimizer.state_dict(),epoch=epoch,
                step=step,seed=seed,world_size=world,source_sha256=source_hash),a.output_dir/'latest_train.tmp')
            (a.output_dir/'latest_train.tmp').replace(a.output_dir/'latest_train.pt')
        if world>1:dist.barrier()
    if sha(source)!=source_hash:raise ValueError('Source checkpoint changed during continuation')
    if rank==0:
        artifacts={p.name:sha(p) for p in a.output_dir.iterdir() if p.suffix in ('.pt','.jsonl') or p.name=='model_config.json'}
        (a.output_dir/'manifest.json').write_text(json.dumps(dict(status='complete',plan_sha256=sha(plan_path),steps=step,artifacts=artifacts),indent=2))
    if world>1:dist.destroy_process_group()
    return 0


if __name__=='__main__':raise SystemExit(main())
