"""Collect answer CE/MSE targets and fit puzzle-disjoint linear and MLP probes."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from hrm import hub
from hrm.paths import repository_root, checkpoint_config
from hrm.training.data import puzzle_directory, freeze_training_seed
from hrm.analyse.dynamics.constraint_readout.run import collect, finite
from hrm.analyse.dynamics.jacobian.models import HRM, load_config, analysis_architecture, unwrap_state_dict
from hrm.analyse.dynamics.jacobian.data import read_test_records, select_board_ids, build_loader
TARGETS={'ce':'Mean reference-digit cross entropy over 81 cells, softmax over digits 1..9.',
         'mse':'Mean squared probability error over 81 cells and nine digit classes.'}


def write_new_json(path,value):
    with Path(path).open('x') as f:json.dump(finite(value),f,indent=2,allow_nan=False)

def answer_errors(full_logits, targets):
    """Return per-puzzle CE and MSE, retaining all 81 cells including clues."""
    if full_logits.ndim != 3 or tuple(full_logits.shape[1:]) != (82, 10):
        raise ValueError('Expected B x 82 x 10 diagnostic logits')
    if targets.shape != full_logits.shape[:2]:
        raise ValueError('Expected B x 82 reference targets including BOS')
    labels = targets[:, 1:] - 1
    if not bool(((labels >= 0) & (labels < 9)).all()):
        raise ValueError('Every reference board digit must be 1..9')
    digits = full_logits[:, 1:, 1:10]
    log_probability = digits.log_softmax(-1)
    ce = -log_probability.gather(-1, labels[..., None]).squeeze(-1).mean(-1)
    probability = digits.softmax(-1)
    one_hot = F.one_hot(labels, num_classes=9).to(probability.dtype)
    mse = (probability - one_hot).square().mean((-2, -1))
    return (ce, mse)

def fit_target(original_code, data, target):
    destination = ROOT / 'probes' / target
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    n, steps, width = data['features'].shape
    a, b = n//2, n//2+n//4
    slices = dict(train=slice(0, a), validation=slice(a, b), test=slice(b, n))
    x = data['features'].float()
    error_tensor = data[target]
    y, delta = (error_tensor.numpy(), data['delta'].numpy())
    flatx, flaty = (x[:a].flatten(0, 1), error_tensor[:a].flatten())
    predictions, fits = ({}, {})
    for label, xx in [('linear_HL', x), ('linear_L', x[..., :width//2]), ('linear_H', x[..., width//2:])]:
        w, bias = original_code.fit_linear(xx[:a].flatten(0, 1), flaty)
        predictions[label] = (xx.double() @ w + bias).numpy()
        fits[label] = dict(weight=w, bias=bias)
    predictions['time_only'] = np.broadcast_to(y[:a].mean(0), y.shape).copy()
    generator = torch.Generator().manual_seed(PLAN["seed"])
    shuffled = error_tensor[:a][torch.randperm(a, generator=generator)].flatten()
    w, bias = original_code.fit_linear(flatx, shuffled)
    predictions['puzzle_shuffled_linear'] = (x.double() @ w + bias).numpy()
    fits['puzzle_shuffled_linear'] = dict(weight=w, bias=bias)
    for seed in [1, 2, 3]:
        head, fitted = original_code.fit_mlp(flatx, flaty, x[a:b].flatten(0, 1), error_tensor[a:b].flatten(), PLAN["mlp_seeds"][seed-1], PLAN["epochs"])
        with torch.no_grad():
            pred = torch.cat([head((chunk - fitted['mean']) / fitted['scale']).squeeze(-1) for chunk in x.flatten(0, 1).split(4096)])
        predictions[f'mlp_seed{seed}'] = (pred * fitted['target_scale'] + fitted['target_mean']).reshape(n, steps).numpy()
        fits[f'mlp_seed{seed}'] = fitted
        print(json.dumps(dict(phase='fit_mlp', target=target, seed=seed, chosen_epoch=fitted['epoch'], seconds=time.monotonic() - started)), flush=True)
    rows = [dict(method=label, split=split, **original_code.summary(y[sl], pred[sl], delta[sl])) for label, pred in predictions.items() for split, sl in slices.items()]
    original_code.common.csvout(destination / 'metrics.csv', rows)
    report = dict(target=target, target_definition=TARGETS[target], metrics=rows,  error_target_next_L=original_code.corr(y[b:], delta[b:]), without_initial_state=original_code.summary(y[b:, 1:], predictions['linear_HL'][b:, 1:], delta[b:, 1:]), bootstrap={}, time_windows={}, normalization_sensitivity=None, omitted_original_branches='Reference correlation and 10-class probability-normalization sensitivity concern the original excess-mass target; both new targets are explicitly fixed to nine digit classes.')
    rng = np.random.default_rng(PLAN["seed"]+4)
    for label in ['linear_HL', 'mlp_seed1', 'time_only']:
        values = []
        for _ in range(PLAN["bootstrap"]):
            idx = rng.integers(b, n, size=n - b)
            values.append([original_code.corr(y[idx], predictions[label][idx]), original_code.corr(predictions[label][idx], delta[idx])])
        report['bootstrap'][label] = dict(zip(['pearson_error_95ci', 'pearson_next_L_95ci'], np.quantile(values, [0.025, 0.975], axis=0).T.tolist()))
    for label in ['linear_HL', 'mlp_seed1', 'time_only']:
        report['time_windows'][label] = {f'{lo}:{hi}': original_code.summary(y[b:, lo:hi], predictions[label][b:, lo:hi], delta[b:, lo:hi]) for lo, hi in [(0, steps // 4), (steps // 4, steps // 2), (steps // 2, steps)]}
    torch.save(fits, destination / 'probes.pt')
    np.savez_compressed(destination / 'predictions.npz', **predictions, true_error=y, next_L=delta, row_ids=data['ids'].numpy(), split=np.r_[np.zeros(a), np.ones(b - a), np.full(n - b, 2)])
    write_new_json(destination / 'results.json', report)
    print(json.dumps({'target':target,'split':[a,b-a,n-b],'complete':True}),flush=True)

def main(argv=None):
    global ROOT, PLAN
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path);p.add_argument('--dataset',type=Path)
    p.add_argument('--output',type=Path,default=repository_root()/'exp/error_readout/records/answer_error')
    p.add_argument('--boards',type=int,default=512);p.add_argument('--cycles',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',default='cpu')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    if a.boards<8 or a.boards%4 or min(a.cycles,a.batch_size,a.epochs,a.bootstrap)<1:p.error('Use >=8 boards divisible by four and positive budgets')
    ROOT=a.output;ROOT.mkdir(parents=True,exist_ok=True)
    seed=freeze_training_seed(ROOT)
    cp=hub.checkpoint('sudoku',a.checkpoint)
    config=load_config(a.config or checkpoint_config(cp)) if a.checkpoint else {'arch':hub.model_config('sudoku')}
    arch=analysis_architecture(config);name=arch.pop('name','hrm@HRM')
    if name not in ('hrm@HRM','hrm.models.hrm@HRM'):p.error('Attention HRM required')
    records=read_test_records(a.dataset or puzzle_directory())
    ids=select_board_ids(len(records),boards=a.boards,offset=0,selection='random',selection_seed=seed,explicit=None)
    old=json.loads((ROOT/'config.json').read_text()) if (ROOT/'config.json').exists() else None
    mlp_seeds=old['mlp_seeds'] if old else []
    while len(mlp_seeds)<3:
        now=int(time.time())
        if now not in mlp_seeds:mlp_seeds.append(now)
        else:time.sleep(.05)
    PLAN=dict(mlp_seeds=mlp_seeds,seed=seed,checkpoint=str(cp),model_config=config,row_ids=list(ids),cycles=a.cycles,epochs=a.epochs,bootstrap=a.bootstrap)
    frozen=ROOT/'config.json'
    if frozen.exists() and json.loads(frozen.read_text())!=PLAN:raise ValueError('Use a new output directory for changed settings')
    frozen.write_text(json.dumps(PLAN,indent=2))
    if a.prepare_only:return
    with (ROOT/'started.json').open('x') as f:json.dump({'seed':seed},f)
    torch.manual_seed(seed);torch.set_num_threads(2)
    model=HRM(arch).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(torch.load(cp,map_location='cpu',weights_only=True)),strict=True)
    data,_=collect(model,build_loader(records,ids,a.batch_size),a.cycles,a.device,answer_targets=True)
    (ROOT/'collection').mkdir();torch.save(data,ROOT/'collection/trajectory.pt')
    from . import reproduce_internal_critic as fitting
    for target in ('ce','mse'):fit_target(fitting,data,target)

if __name__=='__main__':main()
