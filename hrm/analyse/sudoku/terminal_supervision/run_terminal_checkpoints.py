"""Run the existing single-clue evaluation without changing its code or inputs."""
from hrm.paths import repo_path, work_path
from pathlib import Path
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
ROOT = Path(str(work_path('hrm')))
OUT = ROOT / 'outputs/terminal_checkpoints'
GROUPS = {'last_hrm': (3, {f'hrm_s{s}': f'hrm_1k_v2/seed_{s}' for s in (1, 2, 3)}), 'last_rt': (4, {f'rt_s{s}': f'rt_1k_v2/seed_{s}' for s in (1, 2, 3)}), 'last_mae': (5, {f'mae_s{s}': f'mae_flops_matched golden-raccoon/seed_{s}' for s in (1, 2, 3)}), 'last_mae_ds': (6, {f'mae_ds_s{s}': f'mae_ds_flops_matched invincible-crane/seed_{s}' for s in (1, 2, 3)})}

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def worker(group):
    gpu, ckpts = GROUPS[group]
    target = OUT / group
    target.mkdir(parents=True, exist_ok=True)
    cache = OUT / 'runtime' / group
    for part in ('tmp', 'cache', 'triton', 'inductor', 'cuda', 'wandb', 'hf', 'mpl'):
        (cache / part).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(cache / 'tmp'), TMP=str(cache / 'tmp'), TEMP=str(cache / 'tmp'), XDG_CACHE_HOME=str(cache / 'cache'), TRITON_CACHE_DIR=str(cache / 'triton'), TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), CUDA_CACHE_PATH=str(cache / 'cuda'), WANDB_DIR=str(cache / 'wandb'), WANDB_MODE='disabled', HF_HOME=str(cache / 'hf'), MPLCONFIGDIR=str(cache / 'mpl'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    commands = []
    for split in ('train', 'test_hard'):
        cmd = [str(ROOT / '.venv/bin/python'), '-u', 'experiments/examine_failure_mode.py', '--split', split, '--pairs', f'outputs/dataset_size/pairs_{split}.npz', '--batch-size', '256', '--seed', '0', '--num-examples', '0', '--out', str(target)]
        for label, folder in ckpts.items():
            cmd += ['--ckpt', f'{label}=checkpoints/{folder}/last.pt']
        commands.append(cmd)
    (target / 'commands.json').write_text(json.dumps(commands, indent=2))
    codes = []
    for split, cmd in zip(('train', 'test_hard'), commands):
        with open(target / f'{split}.log', 'w') as log:
            code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        codes.append(code)
        (target / 'status.json').write_text(json.dumps({'exit_codes': codes, 'finished_splits': len(codes), 'complete': len(codes) == 2, 'updated_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
        if code:
            raise SystemExit(code)

def launch():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        raise SystemExit('Manifest already exists; inspect existing jobs before relaunching.')
    files = [ROOT / 'experiments/examine_failure_mode.py', ROOT / 'train.py', ROOT / 'arch/mae.py', ROOT / 'arch/mae_deep_supervision.py', ROOT / 'arch/hrm.py', ROOT / 'arch/rt.py', ROOT / 'arch/layers.py']
    files += [ROOT / f'outputs/dataset_size/pairs_{s}.npz' for s in ('train', 'test_hard')]
    for _, ckpts in GROUPS.values():
        for folder in ckpts.values():
            files += [ROOT / 'checkpoints' / folder / f for f in ('last.pt', 'model_config.json')]
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol': 'Exploratory checkpoint-selection sensitivity analysis fixed before this evaluation: use existing terminal EMA last.pt snapshots for all four model families and all seeds 1/2/3. Same archived original/edited puzzle cohorts and final checkpoint-native readout, batch 256. No retraining, seed filtering or checkpoint choice based on these new scores. Terminal snapshots avoid independently selecting each snapshot by peak test_hard score; HRM/RT stopping rules nevertheless use test_hard and the broader comparison is already exploratory. Training recipes and step budgets differ across model families. Do not claim untouched-test evaluation or matched training budgets.', 'groups': GROUPS, 'sha256': {str(p): digest(p) for p in files}, 'pids': {}}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for group in GROUPS:
        with open(OUT / f'{group}_worker.log', 'w') as log:
            proc = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-B', str(Path(__file__).resolve()), group], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        manifest['pids'][group] = proc.pid
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['pids']))
if __name__ == '__main__':
    if len(sys.argv) > 1:
        worker(sys.argv[1])
    else:
        launch()
