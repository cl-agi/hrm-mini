"""Collect Sudoku constraint targets and fit held-out affine H/L readouts."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from hrm import hub
from hrm.paths import repository_root, checkpoint_config
from hrm.training.data import puzzle_directory
from hrm.analyse.dynamics.jacobian.models import HRM, load_config, analysis_architecture, unwrap_state_dict
from hrm.analyse.dynamics.jacobian.data import read_test_records, select_board_ids, build_loader, dataset_sha256
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha


def finite(value):
    if isinstance(value, dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [finite(v) for v in value]
    if isinstance(value, (float,np.floating)) and not np.isfinite(value):return None
    return value


def hard(board, clues):
    pos = torch.arange(81, device=board.device)
    r, c = pos // 9, pos % 9
    peers = ((r[:, None] == r) | (c[:, None] == c) |
             ((r[:, None] // 3 == r // 3) & (c[:, None] // 3 == c // 3)))
    peers.fill_diagonal_(False)
    bad = ((board[..., :, None] == board[..., None, :]) & peers).any(-1)
    clue_bad = (clues > 0) & (board != clues)
    return bad.sum(-1), (bad | clue_bad).sum(-1), clue_bad.sum(-1)


def checks():
    grid = torch.tensor([[(r * 3 + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]).reshape(1, 81)
    assert hard(grid, grid)[0].item() == 0
    altered = grid.clone(); altered[0, 0] = altered[0, 1]
    # Changed cell, row/box peer, and column peer: union is 3 cells.
    assert hard(altered, grid)[0].item() == 3
    assert hard(torch.ones_like(grid), grid)[0].item() == 81
    relabeled = grid % 9 + 1
    a, b, c = hard(relabeled, grid)
    assert (a.item(), b.item(), c.item()) == (0, 81, 81)


def excess_error(prob):
    """Probability-mass excess summed over rows, columns and boxes."""
    p = prob.reshape(*prob.shape[:-2], 9, 9, 9)
    row = (p.sum(-2) - 1).clamp_min(0).sum((-2, -1))
    col = (p.sum(-3) - 1).clamp_min(0).sum((-2, -1))
    boxes = p.reshape(*p.shape[:-3], 3, 3, 3, 3, 9).sum((-4, -2))
    return row + col + (boxes - 1).clamp_min(0).sum((-3, -2, -1))


def fit_linear(x, y):
    # Exact unregularized least squares with intercept; CPU SVD handles rank loss.
    mu, ym = x.double().mean(0), y.double().mean()
    w = torch.linalg.lstsq(x.double()-mu, (y.double()-ym)[:, None], driver='gelsd').solution[:, 0]
    return w, ym-mu@w


@torch.inference_mode()
def collect(model, loader, cycles, device, answer_targets=False):
    parts={name:[] for name in ('features','error','delta','counts','ids')}
    if answer_targets:parts.update(ce=[],mse=[])
    for x, y, ids in loader:
        x=x.to(device);emb=model.embed(x)
        h=model.zH_init.expand_as(emb).clone();l=model.zL_init.expand_as(emb).clone()
        values={name:[] for name in ('features','error','delta','counts')}
        if answer_targets:values.update(ce=[],mse=[])
        for _ in range(cycles):
            for _ in range(model.H_cycles):
                for _ in range(model.L_cycles):
                    feature=torch.cat([l.mean(1),h.mean(1)],-1)
                    logits=model.lm_head(model.H_level(h+l))[:,1:].float()
                    if answer_targets:
                        labels=y[:,1:].to(device)-1
                        prob=logits[...,1:10].softmax(-1)
                        values['ce'].append((-logits[...,1:10].log_softmax(-1).gather(-1,labels[...,None]).squeeze(-1).mean(-1)).cpu())
                        values['mse'].append(((prob-torch.nn.functional.one_hot(labels,9)).square().mean((-1,-2))).cpu())
                    board=logits[...,1:10].argmax(-1)+1
                    nxt=model.L_level(l+h+emb)
                    values['features'].append(feature.cpu())
                    values['error'].append(excess_error(logits[...,1:10].softmax(-1)).cpu())
                    values['delta'].append((nxt-l).float().flatten(1).norm(dim=-1).cpu())
                    values['counts'].append(torch.stack(hard(board,x[:,1:]),-1).cpu())
                    l=nxt
                h=model.H_level(h+l)
        for name in values:parts[name].append(torch.stack(values[name],1))
        parts['ids'].append(ids)
    data={name:torch.cat(values) for name,values in parts.items()}
    return data,{'counts':data.pop('counts'),'collection':'Fresh native recurrence; diagnostic H pass leaves the carry unchanged'}


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a*a).sum() * (b*b).sum()))


def stats(a, b, rng):
    """Bootstrap whole puzzles; permute whole trajectories after time centering."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n, t = a.shape
    # Sufficient moments retain every within-puzzle observation in cluster draws.
    moments = np.stack([a.sum(1), b.sum(1), (a*a).sum(1), (b*b).sum(1), (a*b).sum(1)], 1)
    weights = rng.multinomial(n, np.full(n, 1/n), size=CONFIG['bootstrap_puzzles'])
    sa, sb, aa, bb, ab = (weights @ moments).T
    boot = (ab-sa*sb/(n*t))/np.sqrt((aa-sa*sa/(n*t))*(bb-sb*sb/(n*t)))
    ac, bc = a-a.mean(0), b-b.mean(0)
    denom = np.sqrt((ac*ac).sum()*(bc*bc).sum())
    if denom == 0:
        return {'r': correlation(a,b), 'puzzle_bootstrap_95ci': np.quantile(boot,[.025,.975]).tolist(),
                'within_time_r': None, 'within_time_puzzle_permutation_p_two_sided': None}
    observed = (ac*bc).sum()/denom
    cross = ac @ bc.T
    null = np.array([cross[np.arange(n), rng.permutation(n)].sum()/denom for _ in range(CONFIG['permutation_puzzles'])])
    return {'r': correlation(a,b), 'puzzle_bootstrap_95ci': np.quantile(boot,[.025,.975]).tolist(),
            'within_time_r': float(observed), 'within_time_puzzle_permutation_p_two_sided': float((1+(np.abs(null)>=abs(observed)).sum())/(CONFIG['permutation_puzzles']+1))}


def analyze(data, replay, config, output):
    global CONFIG, OUT
    CONFIG, OUT = config, output
    rng = np.random.default_rng(CONFIG['seed'])
    x = data['features'].double(); n, t, width = x.shape; a, b = n//2, n//2+n//4
    delta = data['delta'].numpy().astype(float)
    targets = {'soft': data['error'].numpy().astype(float),
               'hard': replay['counts'][...,0].numpy().astype(float),
               'hard_including_clues': replay['counts'][...,1].numpy().astype(float)}
    results = {}; arrays = {}; fits = {}
    for name, y in targets.items():
        preds = {}
        for label, xx in [('HL', x), ('L', x[...,:width//2]), ('H', x[...,width//2:])]:
            w, bias = fit_linear(xx[:a].flatten(0,1), torch.from_numpy(y[:a].ravel()))
            preds[label] = (xx@w+bias).numpy()
            fits[name+'_'+label] = {'weight':w,'bias':bias}
        preds['time_only'] = np.broadcast_to(y[:a].mean(0), y.shape)
        results[name] = {}
        for label, pred in preds.items():
            result = {'target': stats(y[b:],pred[b:],rng) if label != 'time_only' else {'r':correlation(y[b:],pred[b:])},
                      'next_L':stats(pred[b:],delta[b:],rng) if label != 'time_only' else {'r':correlation(pred[b:],delta[b:])},
                      'test_R2':float(1-((y[b:]-pred[b:])**2).sum()/((y[b:]-y[b:].mean())**2).sum()),
                      'test_RMSE':float(np.mean((y[b:]-pred[b:])**2)**.5)}
            results[name][label]=result; arrays[name+'_'+label]=pred
            print(json.dumps({'target':name,'probe':label,**result}),flush=True)
        arrays[name+'_true']=y
    results['old_soft_HL_vs_hard'] = stats(targets['hard'][b:],arrays['soft_HL'][b:],rng)
    results['soft_vs_hard'] = stats(targets['soft'][b:],targets['hard'][b:],rng)
    results['clue_violation_observations'] = int((replay['counts'][...,2]>0).sum())
    results['clues_change_hard_count_observations'] = int((replay['counts'][...,0]!=replay['counts'][...,1]).sum())
    results['replay'] = {k:v for k,v in replay.items() if k!='counts'}
    results['config'] = CONFIG
    results['torch_version'] = torch.__version__
    results['split'] = [a,b-a,n-b]; results['observations_per_puzzle']=t
    (OUT/'results.json').write_text(json.dumps(finite(results),indent=2,allow_nan=False)+'\n')
    np.savez_compressed(OUT/'predictions.npz',**arrays,next_L=delta,row_ids=data['ids'].numpy())
    torch.save(fits,OUT/'probes.pt')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path);p.add_argument('--dataset',type=Path)
    p.add_argument('--output-dir',type=Path,default=repository_root()/'exp/error_readout/records/sudoku_answers/hrm/analyse/dynamics/constraint_readout')
    p.add_argument('--seed',type=int);p.add_argument('--boards',type=int,default=512)
    p.add_argument('--cycles',type=int,default=16);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--bootstrap-puzzles',type=int,default=2000);p.add_argument('--permutation-puzzles',type=int,default=4999)
    p.add_argument('--device',default='cuda');p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    if a.boards<8 or a.boards%4 or min(a.cycles,a.batch_size,a.bootstrap_puzzles,a.permutation_puzzles)<1:p.error('Boards must be a multiple of four >=8; all budgets must be positive')
    if a.config and not a.checkpoint:p.error('--config requires --checkpoint')
    out=a.output_dir;old=json.loads((out/'run_config.json').read_text()) if (out/'run_config.json').exists() else None
    seed=a.seed if a.seed is not None else old['seed'] if old else int(time.time())
    cp=hub.checkpoint('sudoku',a.checkpoint)
    config=load_config(a.config or checkpoint_config(cp)) if a.checkpoint else {'arch':hub.model_config('sudoku')}
    arch=analysis_architecture(config)
    if arch.pop('name','') not in ('hrm@HRM','hrm.models.hrm@HRM') or arch.get('is_mlp_mixer',False):p.error('Standard attention HRM required')
    if (arch['seq_len'],arch['vocab_size'])!=(82,10) or arch.get('rope_steps') is not None:p.error('Stationary Sudoku HRM required')
    records=read_test_records(a.dataset or puzzle_directory('sudoku'))
    ids=select_board_ids(len(records),boards=a.boards,offset=0,selection='random',selection_seed=seed,explicit=None)
    if len({records[i].question for i in ids})!=len(ids):p.error('Selected cohort contains duplicate puzzles')
    plan=dict(seed=seed,checkpoint=str(cp),checkpoint_sha256=sha(cp),model_config=config,
        dataset_sha256=dataset_sha256(records),board_ids=list(ids),split=[a.boards//2,a.boards//4,a.boards//4],
        inference_cycles=a.cycles,H_cycles=arch['H_cycles'],L_cycles=arch['L_cycles'],batch_size=a.batch_size,
        bootstrap_puzzles=a.bootstrap_puzzles,permutation_puzzles=a.permutation_puzzles,device=a.device,
        code_sha256=sha(Path(__file__)))
    if old is not None and old!=plan:raise ValueError('Settings changed; use a new output directory')
    if old is None:
        if out.exists() and any(out.iterdir()):raise FileExistsError(out)
        out.mkdir(parents=True,exist_ok=True);(out/'run_config.json').write_text(json.dumps(plan,indent=2))
    if a.prepare_only:return 0
    with (out/'started.json').open('x') as f:json.dump({'started_unix':time.time()},f)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(2);torch.manual_seed(seed);torch.backends.cuda.matmul.allow_tf32=False
    model=HRM(arch).to(a.device).eval().requires_grad_(False)
    model.load_state_dict(unwrap_state_dict(torch.load(cp,map_location='cpu',weights_only=True)),strict=True)
    checks()
    data,replay=collect(model,build_loader(records,ids,a.batch_size),a.cycles,a.device)
    with np.errstate(divide='ignore',invalid='ignore'):analyze(data,replay,plan,out)
    (out/'manifest.json').write_text(json.dumps(dict(status='complete',config=plan,
        outputs={name:sha(out/name) for name in ('results.json','predictions.npz','probes.pt')}),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
