"""Explicitly resume the preserved internal-test queues; validation stays paused."""
from hrm.paths import analysis_directory
from pathlib import Path
import json,os,sys,subprocess,time,sqlite3,traceback
P=analysis_directory(__file__)

def write_status(state,**kw):
 t=P/'status.tmp';t.write_text(json.dumps(dict(state=state,time=time.time(),**kw),indent=2));t.replace(P/'status.json')

if __name__=='__main__':
 try:
  st=json.loads((P/'status.json').read_text())
  for w in st.get('workers',{}).values():
   pid=w['pid'];f=Path(f'/proc/{pid}/cmdline')
   if f.exists():assert str(P).encode() not in f.read_bytes(), 'An existing worker is still running'
  old=P/'resume_launcher_pid.json'
  if old.exists():
   pid=json.loads(old.read_text())['pid'];f=Path(f'/proc/{pid}/cmdline')
   if f.exists():assert str(Path(__file__).resolve().parent / 'resume.py').encode() not in f.read_bytes(), 'Resume launcher already running'
  used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
  assert len(used.splitlines())==8 and all(int(v)<1000 for v in used.splitlines()), 'GPUs occupied; do not overwrite another job'
  old.write_text(json.dumps(dict(pid=os.getpid())))
  libs=[str(x) for b in sys.path if b for x in (Path(b)/'nvidia').glob('*/lib') if x.is_dir()]
  qs=[]
  for model,gpus in [('rt','0,1,2,3'),('hrm','4,5,6,7')]:
   d=P/model;db=sqlite3.connect(d/'queue.sqlite',timeout=120)
   # Interrupted tasks are replayed deterministically; completed tasks are retained.
   db.execute("UPDATE tasks SET status='pending' WHERE status IN ('running','failed')");db.commit();db.close()
   (d/'PAUSE').unlink(missing_ok=True)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,LD_LIBRARY_PATH=':'.join(libs),OMP_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',TORCHINDUCTOR_CACHE_DIR='/tmp/validation_tts_'+model)
   with (d/'workers.log').open('a') as f:
    q=subprocess.Popen([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(d/'worker.py')],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
   qs.append((model,q))
  (P/'PAUSE').unlink(missing_ok=True)
  while True:
   for model,_ in qs:subprocess.run([sys.executable,str(P/model/'summarize.py')],check=True)
   write_status('running',workers={m:dict(pid=q.pid,returncode=q.poll()) for m,q in qs})
   if all(q.poll() is not None for _,q in qs):break
   time.sleep(30)
  assert all(q.returncode==0 for _,q in qs),'Worker failure; see workers.log'
  complete=all(json.loads((P/m/'summary.json').read_text())['complete'] for m,_ in qs)
  write_status('complete' if complete else 'paused',reason='finished' if complete else 'pause_marker')
 except Exception:
  write_status('failed',error=traceback.format_exc());raise
