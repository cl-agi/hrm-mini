"""Shared supplied-checkpoint evaluation CLI; outputs stay outside the release by default."""

def main(default_weight_state="weights"):
    import os,json,sys,time,hashlib,importlib,re
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from pathlib import Path
    import numpy as np
    import torch
    from hrm.evaluation.config import load_config
    from hrm.registry import model_class
    from hrm.evaluation.inference import evaluate_batch
    from hrm.paths import checkpoint_config
    import argparse
    from hrm.paths import repo_path, work_path, repository_root
    parser=argparse.ArgumentParser(description="Evaluate supplied checkpoints with explicit inference cycles, denominator and weight state.")
    parser.add_argument("--checkpoint", "--ckpt", "--checkpoint-dir", "--ckpt-dir", help="Path relative to the release root, or absolute")
    parser.add_argument("--seed", default=1, type=int, help="Original training seed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cycles", type=int, default=16)
    parser.add_argument("--full-dataset", action="store_true", help="Include final partial batch for base Sudoku")
    parser.add_argument("--weight-state", choices=["weights", "ema"], default=default_weight_state)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--data','--dataset-path',type=Path)
    parser.add_argument('--task',choices=['sudoku','maze'],default='sudoku')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    from hrm import hub
    if args.checkpoint is None:
     if args.weight_state!='weights':parser.error('Published checkpoints already contain their selected weights; --weight-state ema requires a full local checkpoint')
     from hrm.evaluation.published import add_arguments,run
     default_parser=argparse.ArgumentParser();add_arguments(default_parser)
     destination=args.output or work_path('checkpoint_evaluation')/('hf_'+args.task)
     command=['--task',args.task,'--output',str(destination/'summary.json'),'--seed',str(args.seed),'--cycles',str(args.cycles),'--device',args.device,'--batch-size','96']
     if args.config:command+=['--config',str(args.config)]
     if args.data:command+=['--data',str(args.data)]
     if args.limit:command+=['--limit',str(args.limit)]
     elif args.task=='sudoku' and not args.full_dataset:command+=['--limit','19968']
     run(default_parser.parse_args(command));return
    ck=Path(args.checkpoint)
    if not ck.is_absolute(): ck=repo_path(ck)
    if ck.is_dir():ck=hub.local_file(ck,'model.pt')
    records=[];relative=None
    if args.config is None:
     try:relative=str(ck.resolve().relative_to(repository_root()))
     except ValueError:pass
     manifest=repo_path("data/manifest.jsonl")
     if manifest.is_file():records=[json.loads(line) for line in manifest.read_text().splitlines()]
    record=next((x for x in records if x["path"]==relative), {"sha256":hashlib.sha256(ck.read_bytes()).hexdigest()})
    out=args.output or work_path("checkpoint_evaluation")/(ck.parent.parent.name+"_"+ck.stem+"_cycles"+str(args.cycles))
    spec=dict(checkpoint=str(ck),sha256=record["sha256"],seed=args.seed,cycles=args.cycles,full_dataset=args.full_dataset,weight_state=args.weight_state,trusted_full_training_checkpoint=args.weight_state=="ema",output=str(out))
    out.mkdir(parents=True,exist_ok=True)
    seed=spec['seed'];torch.manual_seed(seed);np.random.seed(seed%2**32);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    cfg=load_config(ck,args.config,dataset=args.data);split=spec.get('split',cfg.data.eval_split);options=dict(cfg.data.__pydantic_extra__ or {});options.pop('seed',None)
    # Legacy saved configurations include disabled partial-fill options.
    # Only remove provably inactive options for the base Sudoku loader.
    ignored_data_options={}
    if cfg.data.name=='hrm.tasks.sudoku':
     for key,default in [('partial_fill_probability',0.0),('partial_fill_remaining_blanks',None),('partial_fill_on_eval',False)]:
      if key in options:
       if options[key]!=default:raise ValueError(f'Active unsupported data option: {key}={options[key]}')
       ignored_data_options[key]=options.pop(key)
    loader,metadata=importlib.import_module(cfg.data.name).create_dataloader(split,cfg.local_batch_size,rank=0,world_size=1,seed=seed,**options)
    if spec.get('full_dataset'):
     assert cfg.data.name=='hrm.tasks.sudoku', 'Full denominator adapter only validated for base Sudoku'
     loader=torch.utils.data.DataLoader(loader.dataset,batch_size=cfg.local_batch_size,shuffle=False,drop_last=False,collate_fn=loader.collate_fn,num_workers=1,worker_init_fn=loader.worker_init_fn,pin_memory=True)
    mcfg=cfg.model_config_dict(metadata);ident=cfg.arch.name;ident=ident if ident.startswith(('arch.','hrm.')) else 'arch.'+ident
    with torch.device(args.device):model=model_class(ident,mcfg)(mcfg)
    assert hashlib.sha256(ck.read_bytes()).hexdigest()==spec['sha256'], 'Checkpoint hash mismatch'
    # Own archived full training states contain NumPy RNG/optimizer state.
    if spec.get('trusted_full_training_checkpoint'):
     assert hashlib.sha256(ck.read_bytes()).hexdigest()==spec['sha256'], 'Full checkpoint hash mismatch'
     weights=torch.load(ck,map_location='cpu',weights_only=False)
     assert 'format_version' in weights and 'model' in weights and 'optimizer' in weights
    else:
     weights=torch.load(ck,map_location='cpu',weights_only=True)
    if spec.get('weight_state')=='ema':
     assert weights['seed']==spec['seed'], 'Use the original training seed'
     assert weights['model_weight_state']=='raw'
     assert weights['ema_location']=='optimizer.state.*.param_ema'
     raw={k.removeprefix('_orig_mod.'):v for k,v in weights['model'].items()}
     named=list(model.named_parameters());ids=[i for g in weights['optimizer']['param_groups'] for i in g['params']]
     assert len(ids)==len(named) and len(set(ids))==len(ids)
     assert [k for k in raw if k in dict(named)]==[k for k,_ in named]
     for (name,param),pid in zip(named,ids):
      state=weights['optimizer']['state'][pid]
      assert 'param_ema' in state and state['param_ema'].shape==param.shape==raw[name].shape
      raw[name]=state['param_ema']
     weights={'model':raw}
    weights=weights.get('model',weights);model.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in weights.items()},strict=True);model.eval()
    correct=ever=total=0;start=time.time();digest=hashlib.sha256();cy=spec.get('cycles',cfg.cycles_per_data)
    with torch.inference_mode():
     for batch in loader:
      x,y=batch[:2];valid=batch[2].bool() if len(batch)>2 and batch[2].ndim==1 else torch.ones(len(x),dtype=torch.bool);digest.update(x[valid].numpy().tobytes());digest.update(y[valid].numpy().tobytes());result=evaluate_batch(model,x.to(args.device),y.to(args.device),cy)
      if not valid.all():
       result['current_correct']=int((result['predictions'].cpu()[valid]==y[valid]).all(-1).sum());result['total']=int(valid.sum());result['ever_correct']=0
      correct+=result['current_correct'];ever+=result['ever_correct'];total+=result['total']
      if total>=(args.limit or 10**12):break
    expected=spec.get('historical_accuracy_from_filename');report=dict(spec);report.update(ignored_inactive_data_options=ignored_data_options,architecture=mcfg,data=cfg.data.model_dump(),split=split,inference_cycles=cy,total=total,dataset_size=len(loader.dataset),drop_last=loader.drop_last,current_correct=correct,current_accuracy=correct/total,ever_correct=ever if total==len(loader.dataset) and not hasattr(loader.dataset,'q') else None,ordered_input_target_sha256=digest.hexdigest(),duration_seconds=time.time()-start,torch=torch.__version__,deterministic=True,source_config=str(args.config or checkpoint_config(ck)),accuracy_difference=None if expected is None else correct/total-expected,comparison_scope='Saved checkpoint evaluation; filename target is a provenance hint, not proof of checkpoint selection')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__ == "__main__":
    main()
