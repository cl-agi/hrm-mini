from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
"""Fixed L-state displacement interventions on three frozen linear-L probes.

The original 81 states, HRM, covariance, random directions and continuation are
unchanged. Signed displacement replaces relative critic-score calibration.
Negative OLS scores are retained without clipping, shifting or cohort selection.
"""
from hrm.paths import repo_path
from pathlib import Path
import argparse
from datetime import datetime, timezone
import json
import sys
import time
from types import ModuleType
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
BASE = analysis_directory(__file__)
HEX = Path(str(work_path('hrm/analyse/sudoku/core')))
ORIGINAL = HEX / 'artifacts/critic_Lonly_20261004'
ORIGINAL_PROBES = HEX / 'artifacts/internal_critic_20261004'
pass
scripts_package = ModuleType('scripts')
scripts_package.__path__ = [str(HEX / 'scripts')]
sys.modules['scripts'] = scripts_package
from hrm.analyse.dynamics.answer_error import critic_target_dose as dose
from hrm.analyse.dynamics.answer_error import intervene_internal_critic as intervention
TARGETS = ('original_excess', 'ce', 'mse')
SIGNED_DISPLACEMENTS = [-0.03, -0.01, -0.002, 0.002, 0.01, 0.03]
SCORE_TOLERANCE = 0.0002
DISPLACEMENT_TOLERANCE = 2e-07
TOKEN_NORM_TOLERANCE = 2e-06
NATIVE_NORM_MAX_ULP = 4

def now():
    return datetime.now(timezone.utc).isoformat()

def read_json(path):
    return json.loads(Path(path).read_text())

def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def arm_list():
    arms = [dict(module='native', signed_state_displacement=0.0, kind='native', seed=-1, sign=0)]
    for signed_alpha in SIGNED_DISPLACEMENTS:
        arms.append(dict(module='L', signed_state_displacement=signed_alpha, kind='critic', seed=-1, sign=1 if signed_alpha > 0 else -1))
        for seed in range(4):
            for sign in (1, -1):
                arms.append(dict(module='L', signed_state_displacement=signed_alpha, kind='control', seed=seed, sign=sign))
    assert len(arms) == 55
    return arms

def probe_source(target):
    return ORIGINAL_PROBES if target == 'original_excess' else BASE / 'probes' / target

def verify_sources(formal=False):
    freeze = BASE / ('fixed_state_source_freeze.json' if formal else 'source_freeze_v3.json')
    frozen = read_json(freeze)
    for path, expected in frozen['sha256'].items():
        assert dose.common.sha(path) == expected, f'Frozen source changed: {path}'
    if formal:
        assert frozen['sha256'][str(Path(__file__).resolve())] == dose.common.sha(__file__)
        checks = read_json(BASE / 'fixed_state_correctness_checks.json')
        assert checks['status'] == 'passed' and checks['script_sha256'] == dose.common.sha(__file__)
    for target in ('ce', 'mse'):
        source = probe_source(target)
        complete = read_json(source / 'completion.json')
        assert complete['status'] == 'complete' and complete['target'] == target
        for name, expected in complete['files'].items():
            assert dose.common.sha(source / name) == expected
    return freeze

def load_probe(target):
    intervention.SOURCE = probe_source(target)
    probe = intervention.Probe()
    fitted = probe.fits['linear_L']
    probe.w = torch.cat([fitted['weight'], torch.zeros_like(fitted['weight'])])
    probe.b = fitted['bias']
    return probe

def load_context():
    setup = torch.load(dose.OLD / 'setup.pt', map_location='cpu', weights_only=True)
    with np.load(dose.OLD / 'analysis_arrays.npz', allow_pickle=False) as previous:
        selected = np.flatnonzero(previous['critic_immediate'][0] >= 1.0) + 32
    ids = read_json(ORIGINAL / 'protocol.json')['included_ids']
    assert len(ids) == len(set(ids)) == 81 and setup['ids'][selected].tolist() == ids
    assert intervention.DOSES == [0.002, 0.01, 0.03]
    assert dose.EVENTS == ['L'] * 6 + ['H'] + ['L'] * 6 + ['H']
    model, identity = dose.common.model_load('hrm', 1, device='cpu')
    assert model.H_cycles == 2 and model.L_cycles == 6
    assert all((not p.requires_grad and p.device.type == 'cpu' for p in model.parameters()))
    return (setup, selected, ids, model, identity)

def prefix(model, setup, positions, first):
    x, y = (setup['x'][positions], setup['y'][positions])
    carry = dose.common.initial(model, len(x))
    for _ in range(2):
        carry = dose.common.step(model, carry, x, 1)
    old_path = ORIGINAL / f'batch_{first:03d}.pt'
    expected_hash = read_json(ORIGINAL / 'completion.json')['files'][old_path.name]
    assert dose.common.sha(old_path) == expected_hash
    old = torch.load(old_path, map_location='cpu', weights_only=True)
    assert torch.equal(x, old['x']) and torch.equal(y, old['y'])
    assert torch.equal(setup['ids'][positions], old['ids'])
    h, l = (carry['z_H'], carry['z_L'])
    assert torch.equal(torch.cat([l.mean(1), h.mean(1)], -1), old['initial']['feature'][0])
    return (x, y, h, l, old)

def compare_native_initial(initial, old, target):
    keys = ['feature', 'error', 'native_logits', 'decoded_logits']
    if target == 'original_excess':
        keys += ['critic', 'mlp']
    for key in keys:
        assert torch.equal(initial[key], old['initial'][key][0]), f'Native initial {key} changed'

def actual_score(probe, h, l):
    return torch.cat([l.mean(1), h.mean(1)], -1).double() @ probe.w + probe.b

def pulse_cache(l, vectors, w, c0):
    a, b = dose.coefficients(l, vectors['critic'], w)
    result = {}
    for signed_alpha in SIGNED_DISPLACEMENTS:
        sign = 1 if signed_alpha > 0 else -1
        theta = torch.full_like(c0, sign * 2 * np.arcsin(abs(signed_alpha) / 2), dtype=torch.double)
        target = intervention.rotate(l, vectors['critic'], abs(signed_alpha), sign)
        expected = c0 + a * (theta.cos() - 1) + b * theta.sin()
        assert ((expected - c0) * sign > 0).all(), 'Analytical critic displacement has wrong sign'
        result[signed_alpha] = dict(state=target, theta=theta, expected_score=expected)
    return result

def pulse(l, vectors, w, c0, cache, arm):
    if arm['kind'] == 'native':
        zeros = torch.zeros(len(l), dtype=torch.double)
        return (l, dict(target_score=c0, theta=zeros, alpha=zeros, token_norm_relative_error=zeros, token_relative_displacement=torch.zeros(l.shape[:2], dtype=torch.double), token_norm_relative_error_per_token=torch.zeros(l.shape[:2], dtype=torch.double), token_critic_projection_change=torch.zeros(l.shape[:2], dtype=torch.double)))
    q = arm['signed_state_displacement']
    target = cache[q]
    altered = target['state'] if arm['kind'] == 'critic' else dose.score_fixed_control(l, vectors[f"random{arm['seed']}"], w, target['theta'], arm['sign'])
    before, after = (l.double(), altered.double())
    radius = before.norm(dim=-1)
    token_displacement = (after - before).norm(dim=-1) / radius
    desired_displacement = (target['state'].double() - before).norm(dim=-1) / radius
    token_error = (after.norm(dim=-1) - radius).abs() / radius
    alpha = (after - before).flatten(1).norm(dim=-1) / before.flatten(1).norm(dim=-1)
    assert (alpha - abs(q)).abs().max() < DISPLACEMENT_TOLERANCE
    assert (token_displacement - desired_displacement).abs().max() < DISPLACEMENT_TOLERANCE
    assert (token_displacement - abs(q)).abs().max() < DISPLACEMENT_TOLERANCE
    assert token_error.max() < TOKEN_NORM_TOLERANCE and alpha.max() <= 0.2
    return (altered, dict(target_score=target['expected_score'] if arm['kind'] == 'critic' else c0, theta=target['theta'], alpha=alpha, token_norm_relative_error=token_error.max(-1).values, token_relative_displacement=token_displacement, token_norm_relative_error_per_token=token_error, token_critic_projection_change=(after - before) @ w.double()))

def check_score(measured, record, c0, arm):
    error = measured - record['target_score']
    assert error.abs().max() < SCORE_TOLERANCE, ('Score calibration failed', arm, error.abs().max())
    if arm['kind'] == 'critic':
        assert ((measured - c0) * arm['sign'] > 0).all(), ('Actual critic change has wrong sign', arm)
    return error

@torch.inference_mode()
def correctness_checks():
    verify_sources(False)
    setup, selected, ids, model, identity = load_context()
    probes = {target: load_probe(target) for target in TARGETS}
    stats = {target: dict(scores=[], maximum_score_error=0.0, maximum_token_norm_error=0.0, maximum_token_displacement_error=0.0, checked_pulses=0, signed_critic_changes={str(q): [] for q in SIGNED_DISPLACEMENTS}) for target in TARGETS}
    started = time.monotonic()
    for first in range(0, 81, 8):
        positions = selected[first:first + 8]
        x, y, h, l, old = prefix(model, setup, positions, first)
        for target, probe in probes.items():
            initial = dose.initial_measure(model, probe, h, l)
            compare_native_initial(initial, old, target)
            c0 = initial['critic']
            w = probe.w[:512]
            vectors = dose.directions(l, w, setup['covariance']['L'])
            cache = pulse_cache(l, vectors, w, c0)
            row = stats[target]
            row['scores'].extend(c0.tolist())
            for arm in arm_list():
                altered, record = pulse(l, vectors, w, c0, cache, arm)
                score = actual_score(probe, h, altered)
                error = check_score(score, record, c0, arm)
                row['maximum_score_error'] = max(row['maximum_score_error'], float(error.abs().max()))
                row['maximum_token_norm_error'] = max(row['maximum_token_norm_error'], float(record['token_norm_relative_error'].max()))
                row['maximum_token_displacement_error'] = max(row['maximum_token_displacement_error'], float((record['token_relative_displacement'] - abs(arm['signed_state_displacement'])).abs().max()))
                row['checked_pulses'] += len(l)
                if arm['kind'] == 'critic':
                    row['signed_critic_changes'][str(arm['signed_state_displacement'])].extend((score - c0).tolist())
        print(json.dumps(dict(phase='fixed_state_check', states=first + len(x), seconds=time.monotonic() - started)), flush=True)
    for target, row in stats.items():
        row['nonpositive_baseline_count'] = sum((value <= 0 for value in row['scores']))
        row['all_81_original_ids_retained'] = True
        row['all_signed_critic_changes_correct'] = True
    receipt = dict(status='passed', created_utc=now(), script_sha256=dose.common.sha(__file__), checkpoint=identity, n=81, ids=ids, arm_count=55, arms=arm_list(), signed_displacements=SIGNED_DISPLACEMENTS, probe_sources={t: str(probe_source(t)) for t in TARGETS}, targets=stats, score_absolute_tolerance=SCORE_TOLERANCE, displacement_tolerance=DISPLACEMENT_TOLERANCE, token_norm_tolerance=TOKEN_NORM_TOLERANCE, cpu_only=True, threads=4, batch_size=8, dtype='float32', attention='math SDPA', formal_continuation_executed=False, prefix_and_algebra_only=True, baseline_prefix_and_probe_independent_initial_fields_bitwise=True, direction_implementation='Unchanged original covariance-preconditioned tangent and four critic-null random directions, retaining all degeneracy assertions.', source_freeze_sha256=dose.common.sha(BASE / 'source_freeze_v3.json'), negative_scores_retained_without_modification=True, seconds=time.monotonic() - started)
    write_new_json(BASE / 'fixed_state_correctness_checks.json', receipt)
    print(json.dumps(dict(status='passed', targets={t: dict(nonpositive=stats[t]['nonpositive_baseline_count'], max_score_error=stats[t]['maximum_score_error']) for t in TARGETS})), flush=True)

def norm_replay(new, old):
    assert new.dtype == old.dtype == torch.float32
    assert torch.isfinite(new).all() and torch.isfinite(old).all()
    assert (new >= 0).all() and (old >= 0).all()
    ulp = (new.contiguous().view(torch.int32).long() - old.contiguous().view(torch.int32).long()).abs()
    assert ulp.max() <= NATIVE_NORM_MAX_ULP, ('Native norm replay differs', int(ulp.max()))
    return {str(k): int((ulp == k).sum()) for k in range(NATIVE_NORM_MAX_ULP + 1)}

@torch.inference_mode()
def run(target):
    freeze = verify_sources(True)
    campaign_protocol = read_json(BASE / 'fixed_state_protocol.json')
    out = BASE / 'fixed_state' / target
    out.mkdir(parents=True, exist_ok=False)
    setup, selected, ids, model, identity = load_context()
    probe = load_probe(target)
    w, bias = (probe.w, probe.b)
    arms = arm_list()
    started = time.monotonic()
    protocol = dict(campaign_protocol)
    protocol.update(status='frozen_before_this_formal_continuation', criterion=target, probe='linear_L', probe_source=str(probe_source(target)), included_ids=ids, modules=['L'], arms=arms, doses=SIGNED_DISPLACEMENTS, intervention_parameter='signed_state_displacement', events=dose.EVENTS, target='c_after=c0+A*(cos(theta)-1)+B*sin(theta); matched controls retain c0.', fork='after fourth H update (24 L steps), FP32 CPU prefix', continuation='one L pulse then unchanged 12L+2H', negative_scores='Retain every value and every original state; no positivity gate, clipping or intercept shift.', score_absolute_tolerance=SCORE_TOLERANCE, relative_state_displacement_match_tolerance=DISPLACEMENT_TOLERANCE, token_norm_relative_tolerance=TOKEN_NORM_TOLERANCE, max_relative_state_displacement=0.2, inference='CPU FP32 math SDPA; threads4; batch8; no model training.', auxiliary_error_field='initial.error remains original constraint excess. initial.decoded_logits retains the readout required to compute true CE/MSE independently.', source_freeze_sha256=dose.common.sha(freeze), sha256={str(p): dose.common.sha(p) for p in [Path(__file__), probe_source(target) / 'probes.pt', dose.OLD / 'setup.pt', dose.OLD / 'analysis_arrays.npz', BASE / 'fixed_state_protocol.json', BASE / 'fixed_state_correctness_checks.json']})
    write_new_json(out / 'protocol.json', protocol)
    replay_hist = {str(k): 0 for k in range(NATIVE_NORM_MAX_ULP + 1)}
    try:
        for first in range(0, 81, 8):
            positions = selected[first:first + 8]
            x, y, h, l, old = prefix(model, setup, positions, first)
            initial = dose.initial_measure(model, probe, h, l)
            compare_native_initial(initial, old, target)
            c0 = initial['critic']
            native = dose.short_run(model, h, l, x, y, w, bias)
            histogram = norm_replay(native['step_norm'], old['step_norm'][0])
            for k in replay_hist:
                replay_hist[k] += histogram[k]
            assert torch.equal(native['exact'], old['exact'][0]), 'Native final exactness changed'
            if target == 'original_excess':
                assert torch.equal(native['critic'], old['critic'][0]), 'Native original critic trace changed'
            vectors = dose.directions(l, w[:512], setup['covariance']['L'])
            cache = pulse_cache(l, vectors, w[:512], c0)
            observations, records = ([], [])
            for arm_index, arm in enumerate(arms):
                altered, record = pulse(l, vectors, w[:512], c0, cache, arm)
                measured = initial if arm['kind'] == 'native' else dose.initial_measure(model, probe, h, altered)
                record['score_error'] = check_score(measured['critic'], record, c0, arm)
                assert torch.equal(measured['native_logits'], initial['native_logits']), 'L pulse changed instantaneous H output'
                result = native if arm['kind'] == 'native' else dose.short_run(model, h, altered, x, y, w, bias)
                observations.append({key: result[key] for key in ('step_norm', 'critic', 'exact')})
                record['initial'] = measured
                record['measured_score_change'] = measured['critic'] - c0
                records.append(record)
                if arm_index % 18 == 0:
                    print(json.dumps(dict(phase='fixed_state_run', target=target, first=first, arm=arm_index, seconds=time.monotonic() - started)), flush=True)
            batch = {key: torch.stack([value[key] for value in observations]) for key in observations[0]}
            batch['initial'] = {key: torch.stack([r['initial'][key] for r in records]) for key in initial}
            for key in records[0]:
                if key != 'initial':
                    batch[key] = torch.stack([r[key] for r in records])
            batch.update(ids=setup['ids'][positions], x=x, y=y, baseline_critic=c0)
            with (out / f'batch_{first:03d}.pt').open('xb') as stream:
                torch.save(batch, stream)
            print(json.dumps(dict(phase='fixed_state_batch_complete', target=target, completed=first + len(x), total=81, seconds=time.monotonic() - started)), flush=True)
        verify_sources(True)
        write_new_json(out / 'native_replay_audit.json', dict(status='passed', n=81, available_probe_independent_non_norm_fields_bitwise=True, initial_H_logits_unchanged_all_pulses=True, original_excess_probe_trace_bitwise=target == 'original_excess', norm_max_allowed_float32_ulp=NATIVE_NORM_MAX_ULP, norm_ulp_histogram=replay_hist, full_hidden_state_archive_available=False))
        write_new_json(out / 'completion.json', dict(status='complete', criterion=target, n=81, arms=55, trajectories=81 * 55, device='cpu', seconds=time.monotonic() - started, checkpoint=identity, intervention_parameter='signed_state_displacement', script_sha256=dose.common.sha(__file__), source_freeze_sha256=dose.common.sha(freeze), files={p.name: dose.common.sha(p) for p in sorted(out.glob('batch_*.pt'))}, metadata_sha256={p.name: dose.common.sha(p) for p in (out / 'protocol.json', out / 'native_replay_audit.json')}))
    except Exception as error:
        write_new_json(out / 'status.json', dict(status='stopped', target=target, exception_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic() - started, observations_preserved=True))
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'run'])
    parser.add_argument('--target', choices=TARGETS)
    args = parser.parse_args()
    if args.action == 'run' and args.target is None:
        parser.error('--target is required for formal run')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with sdpa_kernel(SDPBackend.MATH):
        if args.action == 'check':
            correctness_checks()
        else:
            run(args.target)
if __name__ == '__main__':
    main()
