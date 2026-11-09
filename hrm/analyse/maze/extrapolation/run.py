"""Run only the three authorized D large-size evaluations on eight idle GPUs."""
from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
from pathlib import Path
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
from datetime import datetime, timezone
ROOT = analysis_directory(__file__)
PYTHON = __import__('sys').executable

def now():
    return datetime.now(timezone.utc).isoformat()

def save(state):
    temp = ROOT / 'status.tmp.json'
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(ROOT / 'status.json')

def main():
    lock = (ROOT / 'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / 'status.json').exists():
        raise FileExistsError('Campaign already exists; inspect its state before any retry')
    if json.loads((ROOT / 'checks.json').read_text())['status'] != 'passed':
        raise ValueError('CPU extension check did not pass')
    state = dict(status='running', started_utc=now(), controller_pid=os.getpid(), host=socket.gethostname(), jobs=[])
    save(state)
    try:
        for seed in (1, 2, 3):
            for gpu in range(8):
                busy = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
                if busy:
                    raise RuntimeError(f'GPU {gpu} is occupied by {busy}; no new evaluation launched')
            name = f'D_2D_seed{seed}'
            output = ROOT / 'results' / name
            command = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8', str(Path(__file__).resolve().parent / 'evaluate.py'), '--seed', str(seed), '--output', str(output)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TORCHINDUCTOR_CACHE_DIR=str(work_path('cache/D_2D')))
            log_path = ROOT / 'logs' / f'{name}.log'
            with log_path.open('x') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                row = dict(id=name, seed=seed, pid=process.pid, status='running', started_utc=now(), command=command, output=str(output), log=str(log_path))
                state['jobs'].append(row)
                save(state)
                print(f'{now()} START {name}', flush=True)
                code = process.wait()
            row.update(status='completed' if code == 0 else 'failed', returncode=code, ended_utc=now())
            save(state)
            if code:
                raise RuntimeError(f'{name} failed; see {log_path}')
            metrics = json.loads((output / 'metrics.json').read_text())['metrics']
            if set(metrics) != {'maze_45', 'maze_50', 'maze_60', 'maze_80'}:
                raise ValueError('Large-size test matrix incomplete')
            if any((value['total'] != 1000 for value in metrics.values())):
                raise ValueError('Incomplete test count')
            print(f'{now()} COMPLETE {name}', flush=True)
        state.update(status='inference_completed', ended_utc=now())
        save(state)
    except BaseException as error:
        state.update(status='stopped', error=f'{type(error).__name__}: {error}', ended_utc=now())
        save(state)
        raise
if __name__ == '__main__':
    main()
