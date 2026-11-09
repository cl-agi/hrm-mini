from hrm.paths import analysis_directory
import sys,os,json,subprocess,pathlib,hashlib,fcntl,datetime,traceback,shutil
P=analysis_directory(__file__);rid=sys.argv[1];M=json.loads((P/'manifest.json').read_text());r=next(x for x in M['runs'] if x['run_id']==rid)
w=P/'runs'/rid;w.mkdir(parents=True,exist_ok=True);lock=(w/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(stage,**kw):
 x=dict(run_id=rid,stage=stage,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw);t=w/'status.tmp';t.write_text(json.dumps(x,indent=2));t.replace(w/'status.json')
try:
 status('preflight');assert shutil.disk_usage('/').free>15*1024**3
 assert json.loads((P/'preflight.json').read_text())['ok']
 for n,h in json.loads((P/'source_sha256.json').read_text()).items():assert hashlib.sha256((P/'source'/n).read_bytes()).hexdigest()==h,n
 for n,h in M['dataset_sha256'].items():assert hashlib.sha256((pathlib.Path(r['config']['data']['dataset_name'])/n).read_bytes()).hexdigest()==h,n
 import torch
 assert torch.cuda.device_count()==8
 ck=w/'checkpoints'/rid/f"seed_{r['seed']}"
 if (w/'training_started.json').exists():raise RuntimeError('Existing attempt; inspect before retry to avoid overwriting evidence')
 env=dict(os.environ,PYTHONPATH=str(P/'source'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='offline',WANDB_DISABLE_CODE='true',WANDB_DIR=str(w),MLP_TASK_NAME=rid,HF_HOME=str(w/'cache/huggingface'),HF_DATASETS_CACHE=str(w/'cache/huggingface/datasets'),XDG_CACHE_HOME=str(w/'cache'),TORCHINDUCTOR_COMPILE_THREADS='2',PYTHONUNBUFFERED='1',TORCHINDUCTOR_CACHE_DIR=str(w/'cache/inductor'),TRITON_CACHE_DIR=str(w/'cache/triton'),TMPDIR=str(w/'cache/tmp'))
 pathlib.Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
 (w/'training_started.json').write_text(json.dumps({'run':r,'torch':torch.__version__,'python':sys.version},indent=2));status('training')
 cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=8','--max-restarts=0',str(P/'source/train.py'),'--config-name',rid,'hydra.job.chdir=false','hydra.run.dir=hydra']
 with (w/'train.log').open('a') as f:subprocess.run(cmd,cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 import yaml
 cfg=yaml.safe_load((ck/'model_config.json').read_text())
 for key,value in r['config'].items():assert cfg[key]==value,key
 records=list(map(json.loads,(ck/'metrics.jsonl').read_text().splitlines()))
 assert len(records)==20 and all(x['total']==20000 for x in records)
 assert [x['step'] for x in records]==list(range(4160,83201,4160))
 assert json.loads((ck/'training_complete.json').read_text())['step']==83200
 groups=json.loads((ck/'optimizer_groups.json').read_text());assert groups['groups']==1
 result={'run_id':rid,'architecture':r['architecture'],'size':r['size'],'seed':r['seed'],'final':records[-1],'best':max(records,key=lambda x:x['exact_match']),'optimizer':groups}
 (w/'result.json').write_text(json.dumps(result,indent=2));status('complete')
except BaseException as e:status('failed',error=str(e),traceback=traceback.format_exc());raise
