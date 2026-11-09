from hrm.paths import repo_path, work_path
from pathlib import Path
from hrm.paths import repo_path
import csv, hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT = Path(str(work_path('mixer_mechanism')))
TRAIN = Path(str(work_path('mixer_reproduce')))
SOURCE = TRAIN / 'source/HRM-Mini-Dev-dev-scaling'
OLD = Path(str(work_path('hrm/analyse/sudoku/core')))
AUDIT = OLD / 'exp/08_h_drive_modulation'
pass
from hrm.training.mixer import TrainConfig
from hrm.training.trainer import load_module

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(seed, fp32=False):
    matches = sorted(repo_path(f'checkpoint/sudoku/mixer_reproduce/seed{seed}/seed_{seed}').glob('*.pt'))
    if len(matches) != 1:
        raise ValueError(f'Expected one selected Mixer checkpoint for seed {seed}, found {len(matches)}')
    ckpt = matches[0]
    config_path = repo_path(f'data/sudoku/seed{seed}/seed_{seed}/model_config.json')
    cfg = TrainConfig(**json.loads(config_path.read_text()))
    model = load_module('arch.hrm@HRM')(cfg.model_config_dict(dict(vocab_size=10, seq_len=82, is_causal=False)))
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    model = model.cuda().eval().requires_grad_(False)
    if fp32:
        model.float()
        model.embed.cast_to = torch.float32
    return (model, dict(checkpoint=str(ckpt), checkpoint_sha256=sha(ckpt), seed=seed, epoch=19, config=json.loads(config_path.read_text())))

def test_data():
    path = repo_path('data/sudoku/sudoku-extreme-1k/test_hard.csv')
    with path.open() as f:
        rows = list(csv.DictReader(f))

    def encode(k):
        return np.array([[0] + [int(v) for v in r[k].replace('.', '0')] for r in rows], dtype=np.int32)
    x, y = (encode('question'), encode('answer'))
    assert x.shape == y.shape == (20000, 82)
    return (x, y, path)

def transition(model, emb, h, l, cycles=2):
    for _ in range(cycles):
        for _ in range(6):
            l = model.L_level(l + h + emb)
        h = model.H_level(h + l)
    return (h, l)

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)
