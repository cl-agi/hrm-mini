"""Resumable, exclusive eight-GPU queue. Never terminate unrelated processes."""
from hrm.paths import repo_path, work_path
from pathlib import Path
import concurrent.futures, fcntl, json, os, subprocess, sys, time, threading
from pathlib import Path
ROOT = Path(str(work_path('mixer_mechanism')))
S = ROOT / 'scripts'
lock = (ROOT / 'runner.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
jobs = []
for seed in (1, 2, 3):
    name = f'taylor_seed{seed}'
    jobs.append((name, ['audit.py', 'taylor', '--seed', str(seed), '--output-dir', str(ROOT / name), '--boards', '64', '--batch-size', '8']))
for part in range(8):
    for seed in (1, 2, 3):
        name = f'triangle_seed{seed}_part{part}'
        jobs.append((name, ['triangle.py', '--seed', str(seed), '--offset', str(part * 1250), '--boards', '1250', '--batch', '16', '--output', str(ROOT / name)]))
for part in range(8):
    for seed in (1, 2, 3):
        name = f'local_seed{seed}_part{part}'
        jobs.append((name, ['audit.py', 'local', '--seed', str(seed), '--output-dir', str(ROOT / name), '--board-offset', str(part * 2500), '--boards', '2500', '--batch-size', '64', '--full-test']))
state = {'host': 'unspecified', 'pid': os.getpid(), 'started': time.time(), 'jobs': {n: {'status': 'queued'} for n, _ in jobs}}
mutex = threading.Lock()

def update(name, **kw):
    with mutex:
        state['jobs'][name].update(kw)
        state['updated'] = time.time()
        tmp = ROOT / 'suite_status.tmp'
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(ROOT / 'suite_status.json')

def free(gpu):
    s = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    mem, util = map(int, s.strip().split(','))
    return mem < 200 and util <= 5
queue = iter(jobs)

def worker(gpu):
    while True:
        with mutex:
            job = next(queue, None)
        if job is None:
            return
        name, cmd = job
        if (ROOT / name / 'manifest.json').exists():
            m = json.loads((ROOT / name / 'manifest.json').read_text())
            if m.get('status') == 'complete':
                update(name, status='complete', resumed=True)
                continue
        if (ROOT / name).exists():
            update(name, status='failed', error='Incomplete output exists; inspect before retry')
            continue
        update(name, status='waiting_for_gpu', gpu=gpu)
        while not free(gpu):
            time.sleep(15)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
        update(name, status='running', gpu=gpu, started=time.time())
        with (ROOT / f'{name}.log').open('a') as f:
            result = subprocess.run([sys.executable, str(S / cmd[0]), *cmd[1:]], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        update(name, status='complete' if result.returncode == 0 else 'failed', returncode=result.returncode, finished=time.time())
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(worker, range(8)))
failed = [n for n, v in state['jobs'].items() if v['status'] != 'complete']
if failed:
    raise RuntimeError(f'Failed jobs: {failed}')
subprocess.run([sys.executable, str(S / 'aggregate.py')], cwd=ROOT, check=True)
state['status'] = 'complete'
(ROOT / 'suite_status.json').write_text(json.dumps(state, indent=2))
