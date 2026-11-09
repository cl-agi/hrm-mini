from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
"""Calibrated critic-fraction interventions with exact-score, matched-dose controls."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend
ROOT = analysis_directory(__file__).parents[0]
pass
from hrm.analyse.dynamics.answer_error.intervene_internal_critic import common, directions, Probe, EVENTS, SOURCE
from hrm.analyse.dynamics.answer_error.reproduce_internal_critic import excess_error
OLD = work_path('critic_intervention')
DOSES = [-0.3, -0.15, -0.05, 0.05, 0.15, 0.3]

def arm_list(modules=('L', 'H')):
    result = [dict(module='native', target_fraction=0.0, kind='native', seed=-1, sign=0)]
    for module in modules:
        for dose in DOSES:
            result.append(dict(module=module, target_fraction=dose, kind='critic', seed=-1, sign=0))
            for k in range(4):
                for sign in [1, -1]:
                    result.append(dict(module=module, target_fraction=dose, kind='control', seed=k, sign=sign))
    return result

def coefficients(s, v, w):
    s = s.double()
    a = s.mean(1) @ w.double()
    b = (s.norm(dim=-1, keepdim=True) * v).mean(1) @ w.double()
    return (a, b)

def target_rotation(s, v, w, delta):
    """Smallest monotone signed rotation solving A(cos t-1)+B sin t=delta."""
    a, b = coefficients(s, v, w)
    delta = delta.double()
    sign = torch.sign(delta)
    limit = 2 * np.arcsin(0.2 / 2)
    grid = torch.linspace(0, limit, 128, dtype=torch.double)[:, None] * sign[None]
    assert torch.all(b[None] * grid.cos() - a[None] * grid.sin() > 0), 'Non-monotone dose branch'
    reached = a * (np.cos(limit) - 1) + sign * b * np.sin(limit)
    assert torch.all(sign * (reached - delta) >= 0), 'Target exceeds predeclared 20% displacement limit'
    lo = torch.zeros_like(delta)
    hi = torch.full_like(delta, limit)
    for _ in range(60):
        mid = (lo + hi) / 2
        angle = mid * sign
        achieved = a * (angle.cos() - 1) + b * angle.sin()
        too_small = sign * (achieved - delta) < 0
        lo = torch.where(too_small, mid, lo)
        hi = torch.where(too_small, hi, mid)
    theta = sign * (lo + hi) / 2
    sd = s.double()
    t = theta[:, None, None]
    altered = (sd * t.cos() + sd.norm(dim=-1, keepdim=True) * v * t.sin()).to(s.dtype)
    return (altered, theta)

def score_fixed_control(s, r, w, theta, sign):
    """Rotate only the subspace orthogonal to w; match every token's chord."""
    sd = s.double()
    u = w.double() / w.double().norm()
    parallel = (sd @ u)[..., None] * u
    perpendicular = sd - parallel
    radius = perpendicular.norm(dim=-1, keepdim=True)
    chord_half = (theta.abs() / 2).sin()[:, None, None] * sd.norm(dim=-1, keepdim=True)
    ratio = chord_half / radius
    assert torch.all(ratio < 1), 'Cannot match chord in critic-null subspace'
    angle = 2 * ratio.asin()
    return (parallel + perpendicular * angle.cos() + sign * radius * r * angle.sin()).to(s.dtype)

def read_score(h, l, w, bias):
    feature = torch.cat([l.mean(1), h.mean(1)], -1)
    return feature.double() @ w + bias

@torch.inference_mode()
def short_run(model, h, l, x, y, w, bias):
    emb = model.embed(x)
    scores = [read_score(h, l, w, bias)]
    norm = []
    for event in EVENTS:
        if event == 'L':
            nxt = model.L_level(l + h + emb)
            diff = nxt - l
            l = nxt
        else:
            nxt = model.H_level(h + l)
            diff = nxt - h
            h = nxt
        norm.append(diff.flatten(1).norm(dim=-1))
        scores.append(read_score(h, l, w, bias))
    pred = model.lm_head(h).argmax(-1)
    return dict(step_norm=torch.stack(norm, 1).cpu(), critic=torch.stack(scores, 1).cpu(), exact=pred[:, 1:].eq(y[:, 1:]).all(-1).cpu(), final_h=h.cpu(), final_l=l.cpu())

@torch.inference_mode()
def initial_measure(model, probe, h, l):
    f = torch.cat([l.mean(1), h.mean(1)], -1)
    score, mlp = probe.scores(f)
    decoded = model.lm_head(model.H_level(h + l)).float()
    return dict(feature=f, critic=score, mlp=mlp, error=excess_error(decoded[:, 1:, 1:].softmax(-1)), native_logits=model.lm_head(h), decoded_logits=decoded)

def prepare(out, probe_name='linear_HL', modules=('L', 'H')):
    old = np.load(OLD / 'analysis_arrays.npz')
    keep = old['critic_immediate'][0] >= 1.0
    ids = old['ids'][keep]
    protocol = dict(status='frozen before calibrated intervention outcomes', source='same previously used 128-puzzle development cohort; 81 baseline-critic>=1 states', included_ids=ids.tolist(), excluded_count=int((~keep).sum()), threshold=1.0, doses=DOSES, arms=arm_list(modules), events=EVENTS, probe=probe_name, modules=list(modules), fork='after fourth H update (24 L steps), FP32 CPU prefix', target='critic_after = (1+q)*critic_before, one pulse per selected module', direction='same training-covariance-preconditioned per-token tangent as previous experiment', control='four random channel directions and both signs; exact per-token norm and critic projection preserved; per-token chord matched to corresponding critic arm', max_relative_state_displacement=0.2, score_absolute_tolerance=0.0002, relative_state_displacement_match_tolerance=2e-07, token_norm_relative_tolerance=2e-06, inference='GPU FP32 math SDPA, TF32 disabled; first batch CPU/GPU baseline comparison; CPU fallback if relative step discrepancy >=0.0005', plot_y='100*(per-puzzle perturbed step norm - native step norm)/native step norm; puzzle mean and 1000-puzzle-bootstrap CI', primary='immediately next L step; actual and matched-control response shown separately', analysis='follow-up descriptive dose response; not a new confirmatory cohort, no p-value selection', interpretation='intervention on critic-associated state directions, not isolation of the critic scalar from all other information', sha256={str(p.relative_to(ROOT)): common.sha(p) for p in [Path(__file__), SOURCE / 'probes.pt', OLD / 'setup.pt', OLD / 'analysis_arrays.npz']})
    common.js(out / 'protocol.json', protocol)
    return protocol

@torch.inference_mode()
def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob('batch_*.pt')):
        raise FileExistsError('Use a new output directory; existing observations are preserved.')
    modules = ('L',) if args.probe == 'linear_L' else ('L', 'H')
    protocol = prepare(out, args.probe, modules)
    arms = arm_list(modules)
    setup = torch.load(OLD / 'setup.pt', weights_only=True)
    old = np.load(OLD / 'analysis_arrays.npz')
    selected = np.flatnonzero(old['critic_immediate'][0] >= 1.0) + 32
    model_cpu, identity = common.model_load('hrm', 1, device='cpu')
    model, _ = common.model_load('hrm', 1, device=args.device)
    probe = Probe()
    if args.probe == 'linear_L':
        fit = probe.fits['linear_L']
        probe.w = torch.cat([fit['weight'], torch.zeros_like(fit['weight'])])
        probe.b = fit['bias']
    w = probe.w
    bias = probe.b
    device = args.device
    started = time.monotonic()
    checked_cpu_gpu = False
    for first in range(0, len(selected), args.batch_size):
        positions = selected[first:first + args.batch_size]
        x, y = (setup['x'][positions], setup['y'][positions])
        carry = common.initial(model_cpu, len(x))
        for _ in range(2):
            carry = common.step(model_cpu, carry, x, 1)
        h, l = (carry['z_H'], carry['z_L'])
        init = initial_measure(model_cpu, probe, h, l)
        c0 = init['critic']
        if args.probe == 'linear_HL':
            np.testing.assert_allclose(c0.numpy(), old['critic_immediate'][0, positions - 32], atol=0.001, rtol=0)
            assert torch.all(c0 >= 1.0)
        else:
            assert torch.all(c0 > 0), 'L-only critic must be positive on the fixed prior cohort'
        native = short_run(model, h.to(device), l.to(device), x.to(device), y.to(device), w.to(device), bias.to(device))
        if not checked_cpu_gpu:
            cpu = short_run(model_cpu, h, l, x, y, w, bias)
            difference = (cpu['step_norm'] - native['step_norm']).abs() / cpu['step_norm'].clamp_min(1e-06)
            maximum = float(difference.max())
            fallback = maximum >= 0.0005
            common.js(out / 'cpu_gpu_check.json', dict(max_relative_step_difference=maximum, max_state_difference=max(float((cpu['final_h'] - native['final_h']).abs().max()), float((cpu['final_l'] - native['final_l']).abs().max())), original_device=args.device, fallback_to_cpu=fallback))
            if fallback:
                device = 'cpu'
                model = model_cpu
                native = cpu
            checked_cpu_gpu = True
        vector = {m: directions(z, w[sl], setup['covariance'][m]) for m, z, sl in [('L', l, slice(0, 512)), ('H', h, slice(512, 1024))] if m in modules}
        outputs = []
        records = []
        for ai, arm in enumerate(arms):
            module, dose = (arm['module'], arm['target_fraction'])
            if module == 'native':
                altered = None
                z = l
                after_h, after_l = (h, l)
                displacement = torch.zeros(len(x), dtype=torch.double)
                norm_error = torch.zeros(len(x), dtype=torch.double)
                theta = torch.zeros(len(x), dtype=torch.double)
                intended = c0
            else:
                z = l if module == 'L' else h
                ww = w[:512] if module == 'L' else w[512:]
                target, theta = target_rotation(z, vector[module]['critic'], ww, c0 * dose)
                if arm['kind'] == 'critic':
                    altered = target
                    intended = c0 * (1 + dose)
                else:
                    altered = score_fixed_control(z, vector[module][f"random{arm['seed']}"], ww, theta, arm['sign'])
                    intended = c0
                displacement = (altered.double() - z.double()).flatten(1).norm(dim=-1) / z.double().flatten(1).norm(dim=-1)
                target_disp = (target.double() - z.double()).flatten(1).norm(dim=-1) / z.double().flatten(1).norm(dim=-1)
                assert (displacement - target_disp).abs().max() < 2e-07
                norm_error = ((altered.double().norm(dim=-1) - z.double().norm(dim=-1)).abs() / z.double().norm(dim=-1)).max(-1).values
                assert norm_error.max() < 2e-06
                after_h, after_l = (h, altered) if module == 'L' else (altered, l)
            measured = initial_measure(model_cpu, probe, after_h, after_l)
            score_error = measured['critic'] - intended
            assert score_error.abs().max() < 0.0002, (arm, score_error.abs().max())
            if module == 'L':
                assert torch.equal(measured['native_logits'], init['native_logits'])
            result = native if module == 'native' else short_run(model, after_h.to(device), after_l.to(device), x.to(device), y.to(device), w.to(device), bias.to(device))
            outputs.append({k: result[k] for k in ['step_norm', 'critic', 'exact']})
            records.append(dict(initial=measured, target_score=intended, score_error=score_error, alpha=displacement, token_norm_relative_error=norm_error, theta=theta))
            if ai % 18 == 0:
                print(json.dumps(dict(first=first, arm=ai, seconds=round(time.monotonic() - started, 1))), flush=True)
        batch = {k: torch.stack([r[k] for r in outputs]) for k in outputs[0]}
        batch['initial'] = {k: torch.stack([r['initial'][k] for r in records]) for k in init}
        for k in ['target_score', 'score_error', 'alpha', 'token_norm_relative_error', 'theta']:
            batch[k] = torch.stack([r[k] for r in records])
        batch.update(ids=setup['ids'][positions], x=x, y=y, baseline_critic=c0)
        torch.save(batch, out / f'batch_{first:03d}.pt')
        print(json.dumps(dict(completed=first + len(x), total=len(selected), device=device, seconds=round(time.monotonic() - started, 1))), flush=True)
    common.js(out / 'completion.json', dict(status='complete', n=len(selected), arms=len(arms), trajectories=len(selected) * len(arms), device=device, seconds=time.monotonic() - started, checkpoint=identity, script_sha256=common.sha(__file__), files={p.name: common.sha(p) for p in sorted(out.glob('batch_*.pt'))}))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/critic_target_dose_20261004')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--probe', choices=['linear_HL', 'linear_L'], default='linear_HL')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with sdpa_kernel(SDPBackend.MATH):
        run(args)
