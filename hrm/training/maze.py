"""Prepare, train and evaluate the E41 mixed-size HRM/RT experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from hrm import hub
from hrm.paths import repository_root
from hrm.training.data import freeze_training_seed
from hrm.training.maze_data import load_rows, create_train_loader


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def input_identity(path=None):
    return {split:dict(ordered_rows_sha256=digest(load_rows(split,path)))
            for split in ('train','test','test_control')}


def source_identity():
    root=repository_root()
    paths=['hrm/training/maze.py','hrm/training/maze_data.py','hrm/training/steps.py',
           'hrm/training/losses.py','hrm/training/optimizer.py','hrm/models/hrm.py',
           'hrm/models/rt.py','hrm/models/layers.py',
           'hrm/analyse/maze/extrapolation/maze_dataset.py',
           'hrm/analyse/maze/rt_extrapolation/maze_metrics.py']
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def prepare(run_dir, family, dataset_path=None):
    run_dir=Path(run_dir).resolve()
    config=json.loads((repository_root()/f'exp/maze/configs/extrapolation/{family.lower()}.json').read_text())
    config['seed']=freeze_training_seed(run_dir)
    config['dataset_path']=str(Path(dataset_path).resolve()) if dataset_path else None
    config['data_identity']=input_identity(dataset_path)
    config['source_identity']=source_identity()
    config['hf_revision']=None if dataset_path else hub.RELEASES['maze'][1]
    target=run_dir/'resolved_config.json'
    if target.exists():
        if json.loads(target.read_text())!=config:
            raise ValueError('Prepared run differs in configuration/code/data; use a new run directory')
    else:
        hub._write(target,lambda p:p.write_text(json.dumps(config,indent=2)+'\n'))
    return config


def read_config(run_dir):
    config=json.loads((Path(run_dir)/'resolved_config.json').read_text())
    if config['source_identity']!=source_identity():
        raise ValueError('Code changed since preparation; prepare a new run directory')
    if config['data_identity']!=input_identity(config['dataset_path']):
        raise ValueError('Data changed since preparation')
    return config


def make_model(config):
    from hrm.models.hrm import HRM
    from hrm.models.rt import RecurrentTransformer
    cls={'HRM':HRM,'RT':RecurrentTransformer}[config['architecture_name']]
    return cls(config['architecture'])


def configure_compiler(config):
    import torch._dynamo.config as dynamo
    dynamo.recompile_limit=config['runtime']['dynamo_recompile_limit']
    dynamo.accumulated_recompile_limit=config['runtime']['dynamo_accumulated_recompile_limit']


def save_weights(model,path):
    import torch
    hub._write(path,lambda p:torch.save(model.state_dict(),p))


def evaluate(model,config,*,device,rank=0,world_size=1,batch_size=8,final=True):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from hrm.analyse.maze.extrapolation.maze_dataset import RepeatedMazeDataset,collate_fn
    from hrm.analyse.maze.rt_extrapolation.maze_metrics import score_predictions,METRIC_VERSION
    model.eval();results={}
    splits=['test','test_control'] if final and config['evaluation']['include_control'] else ['test']
    with torch.inference_mode():
        for split in splits:
            rows=load_rows(split,config['dataset_path'])
            sizes=sorted({len(r['maze']) for r in rows}) if final else [30]
            for size in sizes:
                if split=='test' and size not in config['evaluation']['sizes']:continue
                subset=[r for r in rows if len(r['maze'])==size]
                if len(subset)!=config['evaluation']['examples_per_dataset']:
                    raise ValueError(f'Wrong evaluation denominator: {split}/{size}')
                examples=[dict(question=''.join(r['maze']),answer=''.join(r['solution'])) for r in subset]
                data=RepeatedMazeDataset(examples,[size]*len(examples))
                loader=DataLoader(data,batch_size=batch_size,collate_fn=collate_fn,
                                  sampler=range(rank,len(data),world_size))
                totals=torch.zeros(5,dtype=torch.long,device=device)
                for x,y in loader:
                    inputs=x.numpy();targets=y.numpy();x=x.to(device);carry=model.initial_carry
                    for _ in range(config['evaluation']['inference_cycles']):carry,logits=model(carry,x)
                    scores=score_predictions(inputs,logits.argmax(-1).cpu().numpy(),targets=targets)
                    totals+=torch.tensor([int(scores[k].sum()) for k in ('shortest_path_correct','valid_path_correct','canonical_correct','format_correct')]+[len(y)],device=device)
                if dist.is_initialized():dist.all_reduce(totals)
                correct,valid,canonical,formatted,total=totals.cpu().tolist()
                if total!=len(subset):raise ValueError('Incomplete distributed evaluation')
                results[f'{split}_{size}']=dict(correct=correct,total=total,accuracy=correct/total,
                    valid_path_accuracy=valid/total,exact_match=canonical/total,format_accuracy=formatted/total,
                    metric='shortest_path_accuracy',metric_version=METRIC_VERSION)
    return results


def train(run_dir,config,eval_batch_size):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from hrm.training.optimizer import AdamATan2
    from hrm.training.steps import train_step
    from hrm.training.records import record_evaluation
    training=config['training'];rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if training['lr_min_ratio']!=1.:raise ValueError('E41 uses constant post-warmup learning rate')
    if world!=training['world_size']:raise ValueError('E41 requires 8 ranks, local batch96/global768')
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']));torch.cuda.set_device(device)
    dist.init_process_group('nccl')
    try:
        # Exclusive marker prevents accidental append/retraining in a used directory.
        error=[None]
        if rank==0:
            try:
                with (run_dir/'training_started.json').open('x') as handle:
                    json.dump({'seed':config['seed']},handle)
            except Exception as exc:error[0]=str(exc)
        dist.broadcast_object_list(error,src=0)
        if error[0]:raise RuntimeError(error[0])
        torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
        loader,metadata=create_train_loader(seed=config['seed'],path=config['dataset_path'],
             rank=rank,world_size=world,batch_size=training['local_batch_size'],num_workers=config['runtime']['num_workers'])
        total_steps=len(loader)*training['cycles_per_data']*training['epochs']
        if total_steps!=training['total_optimizer_steps']:raise ValueError('Training update budget differs')
        configure_compiler(config)
        raw=make_model(config).to(device)
        compiled=torch.compile(raw,dynamic=False,fullgraph=True)
        model=DistributedDataParallel(compiled,static_graph=True)
        optim=AdamATan2(model.parameters(),lr=torch.tensor(0.),betas=(training['beta1'],training['beta2']),
                       weight_decay=training['weight_decay'],ema=training['ema'])
        step=0
        for epoch in range(training['epochs']):
            model.train()
            for x,y in loader:
                x=x.to(device);y=y.to(device);carry=raw.initial_carry
                for _ in range(training['cycles_per_data']):
                    step+=1
                    # Published recipes have constant post-warmup learning rates.
                    lr=training['lr']*min(1.,step/training['lr_warmup_steps'])
                    for group in optim.param_groups:group['lr']=torch.tensor(lr,device='cpu')
                    carry,metrics=train_step(model,carry,optim,x,y)
                if rank==0:print(json.dumps(dict(epoch=epoch,step=step,loss=metrics['loss'].item())),flush=True)
            optim.swap_ema()
            try:
                results=evaluate(compiled,config,device=device,rank=rank,world_size=world,
                                 batch_size=eval_batch_size,final=epoch==training['epochs']-1)
                if rank==0:
                    save_weights(raw,run_dir/'last.pt')
                    for split,row in results.items():
                        record_evaluation(run_dir/'evaluation.jsonl',seed=config['seed'],step=step,split=split,
                            correct=row['correct'],total=row['total'],cycles=config['evaluation']['inference_cycles'],weights='EMA')
                    if epoch==training['epochs']-1:
                        hub._write(run_dir/'final_metrics.json',lambda p:p.write_text(json.dumps(dict(epoch=epoch,step=step,weights='EMA',metrics=results),indent=2)+'\n'))
            finally:optim.swap_ema()
            dist.barrier()
    finally:dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','evaluate'))
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--family',choices=('HRM','RT'),help='Used only during prepare; default HRM')
    parser.add_argument('--dataset-path',type=Path,help='Used only during prepare; directory with all three Parquet splits')
    parser.add_argument('--eval-batch-size',type=int,default=8)
    args=parser.parse_args()
    if args.eval_batch_size<1:parser.error('eval-batch-size must be positive')
    if args.action!='prepare' and (args.family is not None or args.dataset_path is not None):
        parser.error('Family and dataset path are frozen during prepare; use a new prepared run to change them')
    if args.action=='prepare':
        config=prepare(args.run_dir,args.family or 'HRM',args.dataset_path)
        print(json.dumps(dict(run_dir=str(args.run_dir),seed=config['seed'],family=config['architecture_name'])))
        return
    config=read_config(args.run_dir)
    if args.action=='train':
        if 'LOCAL_RANK' not in os.environ:parser.error('Launch training with torchrun --nproc-per-node=8')
        train(args.run_dir,config,args.eval_batch_size)
    else:
        import torch
        model=make_model(config).cuda()
        model.load_state_dict(torch.load(args.run_dir/'last.pt',map_location='cpu',weights_only=True),strict=True)
        results=evaluate(model,config,device='cuda',batch_size=args.eval_batch_size)
        output=args.run_dir/'reevaluation.json'
        hub._write(output,lambda p:p.write_text(json.dumps(results,indent=2)+'\n'))
        print(json.dumps(results))


if __name__=='__main__':main()
