"""Queue RT after the complete HRM 1024 stage, never preempt its workers."""
from hrm.paths import analysis_directory
from pathlib import Path
import fcntl,json,shutil,subprocess,sys,time
P=analysis_directory(__file__)
H=P.parent/'sat_public_starts64_outer_20261025_v1'
lock=(P/'handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(state,**extra):
    f=P/'handoff.tmp';f.write_text(json.dumps(dict(state=state,time=time.time(),**extra),indent=2));f.replace(P/'handoff_status.json')
while True:
    if (P/'PAUSE').exists() or (H/'PAUSE').exists():
        status('paused');break
    report=json.loads((H/'stage_report.json').read_text())
    state=json.loads((H/'status.json').read_text())
    if report['stage']==1024 and report['complete'] and all(w['returncode']==0 for w in state['workers']):
        assert shutil.disk_usage('/').free>15*1024**3
        assert shutil.disk_usage(P).free>30*1024**3
        if (P/'launch_receipt.json').exists():
            status('already_launched');break
        with (P/'launcher.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(P/'launch.py')],stdout=log,stderr=subprocess.STDOUT)
        status('launched',pid=child.pid);break
    status('waiting_for_hrm_1024',evaluated=report['evaluated_formulas'],total=report['total'])
    time.sleep(30)
