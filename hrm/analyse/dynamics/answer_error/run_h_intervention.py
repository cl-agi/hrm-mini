"""Frozen H-only counterpart of the completed fixed-state L intervention.

Only this new directory is written. Existing separately fitted H probes are used
without refitting. The original 81 questions and every signed OLS value remain.
"""
from hrm.paths import analysis_directory
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
ROOT = analysis_directory(__file__)
PARENT = ROOT.parent
pass
import hrm.analyse.dynamics.answer_error.run_fixed_state_intervention as inherited
from hrm.analyse.dynamics.answer_error.collect_fit import answer_errors
dose, intervention = (inherited.dose, inherited.intervention)
TARGETS = inherited.TARGETS
DOSES = inherited.SIGNED_DISPLACEMENTS
PROTOCOL = ROOT / 'h_intervention_protocol.json'
CHECKS = ROOT / 'h_correctness_checks.json'
write_json, read_json, now = (inherited.write_new_json, inherited.read_json, inherited.now)

def arms():
    return [dict(arm, module='native' if arm['kind'] == 'native' else 'H') for arm in inherited.arm_list()]

def protocol_document():
    previous = read_json(PARENT / 'fixed_state_protocol.json')
    frozen = read_json(PARENT / 'fixed_state_source_freeze.json')['sha256']
    files = {**frozen, str(Path(__file__).resolve()): dose.common.sha(__file__)}
    for path in (PARENT / 'fixed_state_source_freeze.json', PARENT / 'fixed_state_protocol.json', PARENT / 'collect_fit.py', inherited.ORIGINAL / 'completion.json'):
        files[str(path)] = dose.common.sha(path)
    return dict(status='frozen before H-only intervention outcomes', created_utc=now(), user_authorization='Full H-only intervention for original constraint excess, CE, and MSE.', parent_L_protocol=str(PARENT / 'fixed_state_protocol.json'), targets=dict(previous['targets'], original_excess='Original constraint-excess target, using its separately fitted H-only probe.'), probe_sources=previous['probe_sources'], probe='linear_H', modules=['H'], included_ids=previous['included_ids'], n=81, new_selection=False, probe_refitting=False, signed_state_displacements=DOSES, doses=DOSES, intervention_parameter='signed_state_displacement', arms=arms(), arm_count=55, trajectories_per_target=4455, dose_definition='Per-token and global relative H-state chord displacement abs(q); theta=sign(q)*2asin(abs(q)/2). L is unmodified.', critic_definition='Separately fitted unregularized linear_H: concat(zeros512, weight_H), unchanged intercept. Negative values retained.', covariance='Original setup.pt covariance[H], computed from the original256 training questions and all192 pre-L pooled-H features.', direction=previous['direction'], control=previous['control'], control_replication=previous['control_replication'], checkpoint_sha256=previous['checkpoint_sha256'], fork=previous['fork'], events=previous['events'], continuation='one H pulse then unchanged12L+2H', primary=dict(event_index_zero_based=0, event='L', description='Immediately next L update, matching the completed L-only experiment.'), secondary=dict(event_index_zero_based=6, event='H', description='First subsequent H update, after six ordinary L updates; prespecified secondary endpoint.'), statistics=previous['statistics'], secondary_statistics='Use the same puzzle-paired analysis and corrections for the six doses; identify this endpoint as secondary, not a replacement primary.', target_readout='Uncommitted extra H diagnostic: lm_head(H_level(h+l)); all81 cells, including clues, excluding BOS/class0 for CE/MSE. Same targets as original probes.', additional_native_readout='Also retain direct lm_head(h) logits and all three errors. H pulses can change these immediately; this is a distinct auxiliary readout.', H_score_trace='H-only score must remain exactly constant during each six-L block; H updates occur at event indices6 and13.', device='cpu', dtype='float32', threads=4, batch_size=8, sdpa='math', TF32=False, score_absolute_tolerance=inherited.SCORE_TOLERANCE, relative_state_displacement_match_tolerance=inherited.DISPLACEMENT_TOLERANCE, token_norm_relative_tolerance=inherited.TOKEN_NORM_TOLERANCE, max_relative_state_displacement=0.2, native_comparison='Original probe-independent initial observables and final exactness bitwise; native FP32 step norms within4ULP. New H scores are independently reconstructed, never compared to old L scores.', interpretation=previous['interpretation'], script_sha256=dose.common.sha(__file__), sha256=files)

def verify_sources(require_checks=False):
    inherited.verify_sources(True)
    protocol = read_json(PROTOCOL)
    assert protocol['script_sha256'] == dose.common.sha(__file__), 'H script changed after protocol freeze'
    for path, expected in protocol['sha256'].items():
        assert dose.common.sha(path) == expected, f'Frozen H dependency changed: {path}'
    assert protocol['arms'] == arms() and protocol['doses'] == DOSES
    assert protocol['modules'] == ['H'] and protocol['probe'] == 'linear_H'
    assert protocol['primary']['event_index_zero_based'] == 0
    assert protocol['secondary']['event_index_zero_based'] == 6
    if require_checks:
        receipt = read_json(CHECKS)
        assert receipt['status'] == 'passed' and receipt['script_sha256'] == dose.common.sha(__file__)
        assert receipt['protocol_sha256'] == dose.common.sha(PROTOCOL)
    return protocol

def load_probe(target):
    intervention.SOURCE = inherited.probe_source(target)
    probe = intervention.Probe()
    fitted = probe.fits['linear_H']
    assert fitted['weight'].shape == (512,)
    probe.w = torch.cat([torch.zeros_like(fitted['weight']), fitted['weight']])
    probe.b = fitted['bias']
    return probe

def initial_measure(model, probe, h, l, y):
    result = dose.initial_measure(model, probe, h, l)
    result['answer_ce'], result['answer_mse'] = answer_errors(result['decoded_logits'], y)
    native = result['native_logits'].float()
    result['native_ce'], result['native_mse'] = answer_errors(native, y)
    result['native_error'] = dose.excess_error(native[:, 1:, 1:].softmax(-1))
    assert all((torch.isfinite(value).all() for value in result.values()))
    return result

def compare_native(initial, old, target):
    keys = ['feature', 'error', 'native_logits', 'decoded_logits']
    if target == 'original_excess':
        keys.append('mlp')
    for key in keys:
        assert torch.equal(initial[key], old['initial'][key][0]), f'Native initial {key} changed'

def independent_h_score(probe, h):
    fitted = probe.fits['linear_H']
    return h.mean(1).double() @ fitted['weight'].double() + fitted['bias']

def verify_h_score(probe, h, l, measured):
    rebuilt = independent_h_score(probe, h)
    assert (rebuilt - measured).abs().max() < 1e-08, 'Independent H-only score reconstruction failed'
    assert (inherited.actual_score(probe, h, l) - measured).abs().max() < 1e-08

def check_h_trace(result):
    scores = result['critic']
    assert scores.shape[1] == 15 and dose.EVENTS == ['L'] * 6 + ['H'] + ['L'] * 6 + ['H']
    assert torch.isfinite(scores).all()
    assert torch.equal(scores[:, :7], scores[:, :1].expand(-1, 7)), 'H score changed during first six L updates'
    assert torch.equal(scores[:, 7:14], scores[:, 7:8].expand(-1, 7)), 'H score changed during second six L updates'

def native_run(model, probe, h, l, x, y, old):
    result = dose.short_run(model, h, l, x, y, probe.w, probe.b)
    hist = inherited.norm_replay(result['step_norm'], old['step_norm'][0])
    assert torch.equal(result['exact'], old['exact'][0]), 'Native final exactness changed'
    check_h_trace(result)
    verify_h_score(probe, h, l, result['critic'][:, 0])
    verify_h_score(probe, result['final_h'], result['final_l'], result['critic'][:, -1])
    return (result, hist)

@torch.inference_mode()
def precheck():
    if not PROTOCOL.exists():
        write_json(PROTOCOL, protocol_document())
    protocol = verify_sources()
    assert not CHECKS.exists(), 'Preserve the previous check receipt; do not overwrite it'
    setup, selected, ids, model, identity = inherited.load_context()
    assert ids == protocol['included_ids'] and identity['checkpoint_sha256'] == protocol['checkpoint_sha256']
    stats = {target: dict(scores=[], checked_pulses=0, maximum_score_error=0.0, maximum_token_norm_error=0.0, maximum_token_displacement_error=0.0, native_norm_ulp_histogram={str(k): 0 for k in range(5)}) for target in TARGETS}
    probes = {target: load_probe(target) for target in TARGETS}
    started = time.monotonic()
    for first in range(0, 81, 8):
        x, y, h, l, old = inherited.prefix(model, setup, selected[first:first + 8], first)
        h_before, l_before = (h.clone(), l.clone())
        for target, probe in probes.items():
            row = stats[target]
            initial = initial_measure(model, probe, h, l, y)
            compare_native(initial, old, target)
            _, histogram = native_run(model, probe, h, l, x, y, old)
            for k, count in histogram.items():
                row['native_norm_ulp_histogram'][k] += count
            c0, w = (initial['critic'], probe.w[512:])
            row['scores'].extend(c0.tolist())
            vectors = dose.directions(h, w, setup['covariance']['H'])
            cache = inherited.pulse_cache(h, vectors, w, c0)
            for arm in arms():
                altered, record = inherited.pulse(h, vectors, w, c0, cache, arm)
                score = inherited.actual_score(probe, altered, l)
                verify_h_score(probe, altered, l, score)
                error = inherited.check_score(score, record, c0, arm)
                assert torch.equal(l, l_before) and torch.equal(h, h_before), 'Pulse modified an original state in place'
                row['checked_pulses'] += len(h)
                row['maximum_score_error'] = max(row['maximum_score_error'], float(error.abs().max()))
                row['maximum_token_norm_error'] = max(row['maximum_token_norm_error'], float(record['token_norm_relative_error'].max()))
                row['maximum_token_displacement_error'] = max(row['maximum_token_displacement_error'], float((record['token_relative_displacement'] - abs(arm['signed_state_displacement'])).abs().max()))
        print(json.dumps(dict(phase='H_precheck', completed=first + len(x), total=81, elapsed_seconds=time.monotonic() - started)), flush=True)
    for row in stats.values():
        row['nonpositive_baseline_ids'] = [i for i, value in zip(ids, row['scores']) if value <= 0]
        row['all_signed_changes_correct'] = True
    verify_sources()
    receipt = dict(status='passed', created_utc=now(), script_sha256=dose.common.sha(__file__), protocol_sha256=dose.common.sha(PROTOCOL), checkpoint=identity, n=81, ids=ids, arm_count=55, targets=stats, cpu_only=True, device='cpu', dtype='float32', threads=4, batch_size=8, sdpa='math', all81_retained=True, negative_scores_unchanged=True, L_unchanged_and_original_H_not_mutated=True, native_probe_independent_non_norm_fields_bitwise=True, native_norm_max_allowed_float32_ulp=4, native_H_only_score_constant_during_both_six_L_blocks=True, independent_H_score_reconstruction=True, inherited_target_definition_checks=str(PARENT / 'target_definition_tests.json'), altered_state_continuation_executed=False, native_continuation_executed=True, seconds=time.monotonic() - started)
    write_json(CHECKS, receipt)
    print(json.dumps(dict(status='passed', seconds=receipt['seconds'], targets=stats)), flush=True)

@torch.inference_mode()
def run(target):
    campaign = verify_sources(require_checks=True)
    output = ROOT / 'fixed_state_H' / target
    output.mkdir(parents=True, exist_ok=False)
    setup, selected, ids, model, identity = inherited.load_context()
    assert ids == campaign['included_ids'] and identity['checkpoint_sha256'] == campaign['checkpoint_sha256']
    probe = load_probe(target)
    protocol = dict(campaign, criterion=target, probe_source=str(inherited.probe_source(target)), parent_protocol_sha256=dose.common.sha(PROTOCOL), checks_sha256=dose.common.sha(CHECKS), primary_event_index=0, secondary_event_index=6)
    write_json(output / 'protocol.json', protocol)
    histogram = {str(k): 0 for k in range(5)}
    started = time.monotonic()
    try:
        for first in range(0, 81, 8):
            positions = selected[first:first + 8]
            x, y, h, l, old = inherited.prefix(model, setup, positions, first)
            h_before, l_before = (h.clone(), l.clone())
            initial = initial_measure(model, probe, h, l, y)
            compare_native(initial, old, target)
            native, hist = native_run(model, probe, h, l, x, y, old)
            for key in hist:
                histogram[key] += hist[key]
            c0, w = (initial['critic'], probe.w[512:])
            vectors = dose.directions(h, w, setup['covariance']['H'])
            cache = inherited.pulse_cache(h, vectors, w, c0)
            observations, records = ([], [])
            for arm_index, arm in enumerate(arms()):
                altered, record = inherited.pulse(h, vectors, w, c0, cache, arm)
                measured = initial if arm['kind'] == 'native' else initial_measure(model, probe, altered, l, y)
                verify_h_score(probe, altered, l, measured['critic'])
                record['score_error'] = inherited.check_score(measured['critic'], record, c0, arm)
                assert torch.equal(l, l_before) and torch.equal(h, h_before), 'Original H or L modified in place'
                result = native if arm['kind'] == 'native' else dose.short_run(model, altered, l, x, y, probe.w, probe.b)
                check_h_trace(result)
                verify_h_score(probe, result['final_h'], result['final_l'], result['critic'][:, -1])
                observations.append({key: result[key] for key in ('step_norm', 'critic', 'exact')})
                record['initial'] = measured
                record['measured_score_change'] = measured['critic'] - c0
                record['native_logits_change'] = measured['native_logits'].float() - initial['native_logits'].float()
                records.append(record)
                if arm_index % 18 == 0:
                    print(json.dumps(dict(phase='H_formal', target=target, first=first, arm=arm_index, seconds=time.monotonic() - started)), flush=True)
            batch = {key: torch.stack([value[key] for value in observations]) for key in observations[0]}
            batch['initial'] = {key: torch.stack([row['initial'][key] for row in records]) for key in initial}
            for key in records[0]:
                if key != 'initial':
                    batch[key] = torch.stack([row[key] for row in records])
            batch.update(ids=setup['ids'][positions], x=x, y=y, baseline_critic=c0)
            with (output / f'batch_{first:03d}.pt').open('xb') as stream:
                torch.save(batch, stream)
            print(json.dumps(dict(phase='H_batch_complete', target=target, completed=first + len(x), total=81, seconds=time.monotonic() - started)), flush=True)
        verify_sources(require_checks=True)
        write_json(output / 'native_replay_audit.json', dict(status='passed', n=81, available_probe_independent_non_norm_fields_bitwise=True, L_unchanged_all_pulses=True, original_H_not_modified_in_place=True, H_score_independently_reconstructed=True, H_score_constant_during_L_blocks_all_pulses=True, initial_H_logits_unchanged_all_pulses=False, initial_H_logits_note='Not required: H interventions may immediately change the direct output.', norm_max_allowed_float32_ulp=4, norm_ulp_histogram=histogram, full_hidden_state_archive_available=False))
        write_json(output / 'completion.json', dict(status='complete', criterion=target, n=81, arms=55, trajectories=4455, device='cpu', checkpoint=identity, intervention_parameter='signed_state_displacement', probe='linear_H', modules=['H'], primary_event_index=0, secondary_event_index=6, seconds=time.monotonic() - started, script_sha256=dose.common.sha(__file__), parent_protocol_sha256=dose.common.sha(PROTOCOL), checks_sha256=dose.common.sha(CHECKS), files={path.name: dose.common.sha(path) for path in sorted(output.glob('batch_*.pt'))}, metadata_sha256={path.name: dose.common.sha(path) for path in (output / 'protocol.json', output / 'native_replay_audit.json')}))
    except Exception as error:
        write_json(output / 'status.json', dict(status='stopped', target=target, exception_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic() - started, observations_preserved=True))
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precheck-only', action='store_true')
    parser.add_argument('--target', choices=TARGETS)
    args = parser.parse_args()
    if args.precheck_only == (args.target is not None):
        parser.error('Choose exactly one of --precheck-only or --target TARGET')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with sdpa_kernel(SDPBackend.MATH):
        if args.precheck_only:
            precheck()
        else:
            run(args.target)
if __name__ == '__main__':
    main()
