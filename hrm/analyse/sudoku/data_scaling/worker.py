from hrm.paths import analysis_directory
import shutil,sys,os,json,subprocess,pathlib,hashlib,fcntl,datetime,traceback
p=analysis_directory(__file__);rid=sys.argv[1];m=json.loads((p/'manifest.json').read_text());r=next(r for r in m['runs'] if r['run_id']==rid);w=p/'runs'/rid;w.mkdir(parents=True,exist_ok=True)
lock=(w/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'run_id':rid,'task_id':os.environ.get('MLP_TASK_ID')}
def status(stage,**kw):
 state.update(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw);(w/'status.json').write_text(json.dumps(state,indent=2))
def execute(cmd,name,env):
 with (w/name).open('a') as f:subprocess.run(cmd,cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 status('preflight')
 assert shutil.disk_usage('/').free > 15*1024**3, 'Root free space below 15GiB buffer'
 assert json.loads((p/'preflight.json').read_text())['ok']
 for name,digest in json.loads((p/'source_sha256.json').read_text()).items():assert hashlib.sha256((p/'source'/name).read_bytes()).hexdigest()==digest,name
 for name,digest in json.loads((p/r['data_dir']/'manifest.json').read_text())['hashes'].items():assert hashlib.sha256((p/r['data_dir']/name).read_bytes()).hexdigest()==digest,name
 import torch
 assert torch.cuda.device_count()==8
 env=dict(os.environ,PYTHONPATH=str(p/'source'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='offline',WANDB_DISABLE_CODE='true',WANDB_DIR=str(w),MLP_TASK_NAME=rid,HF_HOME=str(p/'cache/huggingface'),HF_DATASETS_CACHE=str(p/'cache/huggingface/datasets'),XDG_CACHE_HOME=str(w/'cache'),TORCHINDUCTOR_COMPILE_THREADS='2',PYTHONUNBUFFERED='1')
 env.update(TORCHINDUCTOR_CACHE_DIR=str(w/'cache/inductor'),TRITON_CACHE_DIR=str(w/'cache/triton'),TMPDIR=str(w/'cache/tmp'))
 pathlib.Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
 command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8','--max-restarts=0']
 ck=w/'checkpoints'/rid/f"seed_{r['seed']}";summary=ck/'training_summary.json'
 if not summary.exists():
  if (w/'training_started.json').exists():raise RuntimeError('Partial training exists; inspect before retry')
  (w/'training_started.json').write_text(json.dumps(r));status('training')
  execute(command+[str(p/'source/train.py'),'--config-name',rid,'hydra.job.chdir=false','hydra.run.dir=hydra'],'train.log',env)
 import yaml
 saved=yaml.safe_load((ck/'model_config.json').read_text());expected=json.loads((p/'validation'/f'{rid}.json').read_text());assert saved==expected,'Resolved training configuration mismatch'
 s=json.loads(summary.read_text());assert s['complete'] and s['epochs']==20
 records=[json.loads(l) for l in (ck/'eval_metrics.jsonl').read_text().splitlines()];assert len(records)==20 and all(x['total']==20000 and x['eval_cycles']==128 for x in records)
 assert s['best']==max(records,key=lambda x:(x['exact_match'],x['step']))
 if r['mode']=='regular':assert s['optimizer_steps']==r['train_batches_per_epoch']*(sum(r['config']['outer_schedule']) if r['config']['outer_schedule'] else r['config']['cycles_per_data']*20)
 assert (ck/s['best']['checkpoint']).is_file()
 status('evaluating');execute(command+[str(p/'source/evaluate_best.py'),str(ck)],'eval.log',env)
 result=json.loads((w/'evaluation/result.json').read_text());assert result['complete'] and result['total']==20000
 (w/'result.json').write_text(json.dumps({'run':r,'training':s,'evaluation':result},indent=2));status('complete')
except BaseException as error:
 status('failed',error=str(error),traceback=traceback.format_exc());raise
