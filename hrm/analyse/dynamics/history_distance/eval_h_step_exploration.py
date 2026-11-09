"""Paired H-vs-L historical novelty, with common-history and step-length controls."""
import argparse
import hashlib
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from hrm import hub
from hrm.paths import repository_root, checkpoint_config
from hrm.training.data import puzzle_directory
from hrm.analyse.dynamics.jacobian.data import read_test_records, select_board_ids, build_loader, dataset_sha256
from hrm.analyse.dynamics.jacobian.models import HRM, load_config, analysis_architecture, unwrap_state_dict
from hrm.analyse.dynamics.output_sensitivity.local_analysis import sha as sha256
from hrm.analyse.sudoku.continuous_inference.runner_snapshot import sudoku_valid
from hrm.analyse.dynamics.answer_direction.train_secondary_hl_readout import model_hash

def rms(x):
    return x.square().mean(-1).sqrt()

def distances(q, history):
    return torch.cat([rms(q[:, None] - h) for h in history.split(32, dim=1)], dim=1)

def metrics(a, b, origin, history, bandwidth):
    sa, sb = (rms(a - origin), rms(b - origin))
    da, db = (distances(a, history), distances(b, history))
    radius = torch.minimum(sa, sb)
    am = origin + (a - origin) * (radius / sa.clamp_min(1e-12))[:, None]
    bm = origin + (b - origin) * (radius / sb.clamp_min(1e-12))[:, None]
    dma, dmb = (distances(am, history), distances(bm, history))
    out = {}
    for label, step, d, dm in [('H', sa, da, dma), ('L', sb, db, dmb)]:
        out[label + '_step'] = step
        out[label + '_nn'] = d.min(1).values
        out[label + '_knn5'] = d.topk(min(5, d.shape[1]), largest=False).values.mean(1)
        out[label + '_escape'] = d.min(1).values / step.clamp_min(1e-12)
        out[label + '_matched_nn'] = dm.min(1).values
        out[label + '_matched_escape'] = dm.min(1).values / radius.clamp_min(1e-12)
        for factor in [0.5, 1.0, 2.0]:
            out[label + f'_occupancy_{factor:g}'] = (d <= (bandwidth * factor)[:, None]).float().mean(1)
        out[label + '_matched_occupancy'] = (dm <= bandwidth[:, None]).float().mean(1)
    out['matched_radius'] = radius
    return out

def save_csv(path, rows):
    with open(path, 'w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def summarize(rows, budget, seed, repeats=5000):
    rng = np.random.default_rng(seed)
    result = []
    names = ['nn', 'knn5', 'escape', 'matched_nn', 'matched_escape', 'occupancy_0.5', 'occupancy_1', 'occupancy_2', 'matched_occupancy']
    for l in sorted({r['L_steps'] for r in rows}):
        for probe in ['direct_sum', 'response_L']:
            for window in ['equal_H_cycles', 'equal_compute']:
                rr = [r for r in rows if r['L_steps'] == l and r['probe'] == probe and (r['H_cycle'] > 1) and (window == 'equal_H_cycles' or r['microstep'] <= budget)]
                if not rr:
                    continue
                ids = sorted({r['board_id'] for r in rr})
                for name in names:
                    vals = np.array([[np.mean([r[f'{s}_{name}'] for r in rr if r['board_id'] == i]) for s in ['H', 'L']] for i in ids])
                    delta = vals[:, 0] - vals[:, 1]
                    boot = delta[rng.integers(0, len(ids), (repeats, len(ids)))].mean(1)
                    lo, hi = np.quantile(boot, [0.025, 0.975])
                    result.append(dict(L_steps=l, probe=probe, window=window, metric=name, boards=len(ids), paired_boundaries=len(rr), H_mean=float(vals[:, 0].mean()), L_mean=float(vals[:, 1].mean()), difference=float(delta.mean()), ci_low=float(lo), ci_high=float(hi), board_H_greater_fraction=float((delta > 0).mean())))
    return result

@torch.inference_mode()
def run(a, model, x, y, ids, plan):
    torch.set_num_threads(a.threads)
    out = a.output_dir
    before_hash = model_hash(model)
    rows = []
    performance = []
    final_states = []
    start = time.monotonic()
    for begin in range(0, len(x), a.batch_size):
        xx = torch.tensor(x[begin:begin + a.batch_size], device=a.device)
        yy = torch.tensor(y[begin:begin + a.batch_size], device=a.device)
        emb = model.embed(xx)
        h0 = model.zH_init.expand_as(emb).clone()
        l0 = model.zL_init.expand_as(emb).clone()
        native, _ = model(dict(z_H=h0.clone(), z_L=l0.clone()), xx)
        hh, ll = h0, l0
        for _ in range(2):
            for _ in range(6):
                ll = model.L_level(ll + hh + emb)
            hh = model.H_level(hh + ll)
        if not torch.equal(hh, native['z_H']) or not torch.equal(ll, native['z_L']):
            raise RuntimeError('Native FP32 2H6L parity failed')
        b = len(xx)
        d = emb[0].numel()
        for lsteps in a.l_steps:
            h = model.zH_init.expand_as(emb).clone()
            z = model.zL_init.expand_as(emb).clone()
            lh = torch.empty(b, 1 + a.h_cycles * lsteps, d, device=a.device)
            sh = torch.empty(b, 1 + a.h_cycles * (lsteps + 1), d, device=a.device)
            lh[:, 0] = z.flatten(1)
            sh[:, 0] = (h + z).flatten(1)
            nl = ns = 1
            scales = []
            ever = torch.zeros(b, dtype=torch.bool, device=a.device)
            for cycle in range(1, a.h_cycles + 1):
                for _ in range(lsteps):
                    zn = model.L_level(z + h + emb)
                    if cycle == 1 and len(scales) < model.L_cycles:
                        scales.append(rms((zn - z).flatten(1)))
                    z = zn
                    lh[:, nl] = z.flatten(1)
                    nl += 1
                    sh[:, ns] = (h + z).flatten(1)
                    ns += 1
                if cycle == 1:
                    bandwidth = torch.stack(scales).median(0).values.clamp_min(1e-06)
                hn = model.H_level(h + z)
                extra_l = model.L_level(z + h + emb)
                direct = metrics((hn + z).flatten(1), (h + extra_l).flatten(1), (h + z).flatten(1), sh[:, :ns], bandwidth)
                response_h = model.L_level(z + hn + emb)
                response_l = model.L_level(extra_l + h + emb)
                response = metrics(response_h.flatten(1), response_l.flatten(1), z.flatten(1), lh[:, :nl], bandwidth)
                for probe, values in [('direct_sum', direct), ('response_L', response)]:
                    for j in range(b):
                        rows.append(dict(board_id=int(ids[begin + j]), L_steps=lsteps, H_cycle=cycle, microstep=cycle * (lsteps + 1), probe=probe, bandwidth=float(bandwidth[j]), **{k: float(v[j]) for k, v in values.items()}))
                h = hn
                sh[:, ns] = (h + z).flatten(1)
                ns += 1
                pred = model.lm_head(h).argmax(-1)
                valid = sudoku_valid(pred, xx)
                ever |= valid
                for j in range(b):
                    performance.append(dict(board_id=int(ids[begin + j]), L_steps=lsteps, H_cycle=cycle, microstep=cycle * (lsteps + 1), valid=bool(valid[j]), ever_valid=bool(ever[j]), exact=bool(pred[j, 1:].eq(yy[j, 1:]).all())))
            final_states.append(dict(board_ids=[int(i) for i in ids[begin:begin+b]], L_steps=lsteps, sha256=hashlib.sha256(h.cpu().numpy().tobytes()+z.cpu().numpy().tobytes()).hexdigest()))
            print(json.dumps(dict(completed_boards=begin + b, L_steps=lsteps, elapsed_seconds=round(time.monotonic() - start, 1))), flush=True)
            save_csv(out / 'boundaries.csv', rows)
            save_csv(out / 'performance.csv', performance)
    summary = summarize(rows, a.h_cycles * (min(a.l_steps) + 1), plan['seed'], a.bootstrap_repeats)
    save_csv(out / 'summary.csv', summary)
    if model_hash(model) != before_hash or any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise RuntimeError('Frozen backbone changed')
    report = dict(plan, status='complete', elapsed_seconds=time.monotonic()-start,
        unchanged_model_state_hash=before_hash, native_fp32_parity=True,
        boundary_rows=len(rows), performance_rows=len(performance), final_states=final_states,
        artifact_sha256={name:sha256(out/name) for name in ('cohort.npz','boundaries.csv','performance.csv','summary.csv')},
        notes=['FP32 inference, including embedding and initial carry',
               f'All 82 tokens x {model.zH_init.numel()} features; exact RMS distance',
               'First H phase excluded from summaries; no success-based selection or stopping',
               'H->L and L->L each use two module calls; probes never enter carry/history',
               'Equal-compute window counts scheduled calls only, excluding diagnostic probes; last complete H boundary',
               'Per-board means then paired board bootstrap; one checkpoint, not independent training repeats'])
    (out/'metadata.json').write_text(json.dumps(report,indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path);p.add_argument('--dataset',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/dynamics/records/history_distance')))
    p.add_argument('--seed',type=int);p.add_argument('--boards',type=int,default=64);p.add_argument('--board-ids',type=int,nargs='+')
    p.add_argument('--h-cycles',type=int,default=32);p.add_argument('--l-steps',type=int,nargs='+',default=[6,9,12,18,24,36,48])
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--bootstrap-repeats',type=int,default=5000);p.add_argument('--device',default='cuda')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    if min(a.boards,a.batch_size,a.threads,a.bootstrap_repeats,*a.l_steps)<1 or a.h_cycles<2 or len(set(a.l_steps))!=len(a.l_steps):p.error('Invalid sizes/schedule (at least two H phases required)')
    if a.config and not a.checkpoint:p.error('--config requires --checkpoint')
    out=a.output_dir;old=json.loads((out/'run_config.json').read_text()) if (out/'run_config.json').exists() else None
    seed=a.seed if a.seed is not None else old['seed'] if old else int(time.time())
    cp=hub.checkpoint('sudoku',a.checkpoint)
    config=load_config(a.config or checkpoint_config(cp)) if a.checkpoint else {'arch':hub.model_config('sudoku')}
    arch=analysis_architecture(config)
    if arch.pop('name','') not in ('hrm@HRM','hrm.models.hrm@HRM') or arch.get('is_mlp_mixer',False):p.error('Standard attention HRM required')
    if (arch['seq_len'],arch['vocab_size'],arch.get('H_cycles'),arch.get('L_cycles'))!=(82,10,2,6) or arch.get('rope_steps') is not None:p.error('Stationary Sudoku 2H6L required')
    records=read_test_records(a.dataset if a.dataset else puzzle_directory('sudoku'))
    ids=select_board_ids(len(records),boards=a.boards,offset=0,selection='random',selection_seed=seed,explicit=a.board_ids)
    x,y,_=next(iter(build_loader(records,ids,len(ids))));x=x.numpy();y=y.numpy()
    plan=dict(seed=seed,checkpoint=str(cp),checkpoint_sha256=sha256(cp),config=config,dataset_sha256=dataset_sha256(records),
        board_ids=list(ids),h_cycles=a.h_cycles,l_steps=a.l_steps,batch_size=a.batch_size,threads=a.threads,
        bootstrap_repeats=a.bootstrap_repeats,device=a.device,precision='float32',code_sha256=sha256(Path(__file__)))
    if old is not None and old!=plan:raise ValueError('Changed inputs/settings; use new output')
    if old is None:
        if out.exists() and any(out.iterdir()):raise FileExistsError(out)
        out.mkdir(parents=True,exist_ok=True);(out/'run_config.json').write_text(json.dumps(plan,indent=2))
    if a.prepare_only:return 0
    with (out/'started.json').open('x') as f:json.dump(dict(started_unix=time.time()),f)
    torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False
    model=HRM(arch).to(a.device).eval()
    model.load_state_dict(unwrap_state_dict(torch.load(cp,map_location='cpu',weights_only=True)),strict=True)
    model.float();model.embed.cast_to=torch.float32;model.requires_grad_(False)
    np.savez_compressed(out/'cohort.npz',inputs=x,targets=y,ids=ids)
    run(a,model,x,y,ids,plan)
    return 0


if __name__=='__main__':raise SystemExit(main())
