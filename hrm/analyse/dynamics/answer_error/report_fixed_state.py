from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
"""Audit actual answer errors and summarize the approved fixed-state experiment."""
from hrm.paths import repo_path, work_path
from pathlib import Path
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import torch
ROOT = analysis_directory(__file__)
OUT = ROOT / 'fixed_state'
HEX = Path(str(work_path('cl_agi_hrm_mini/artifacts')))
NAMES = ['original_excess', 'ce', 'mse']
LABEL = {'original_excess': 'Original E', 'ce': 'Answer CE', 'mse': 'Answer MSE'}

def read(path):
    return json.loads(path.read_text())

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    torch.set_num_threads(4)
    frozen = read(ROOT / 'fixed_state_protocol.json')
    results = {}
    for name in NAMES:
        directory = OUT / name
        done = read(directory / 'completion.json')
        protocol = read(directory / 'protocol.json')
        status = read(directory / 'plotting_status.json')
        audit = read(directory / 'plotting_audit.json')
        assert done['status'] == 'complete' and status['status'] == 'completed' and audit['passed']
        assert done['criterion'] == protocol['criterion'] == status['target'] == audit['target'] == name
        assert done['intervention_parameter'] == protocol['intervention_parameter'] == 'signed_state_displacement'
        assert status['protocol'] == 'fixed_state_displacement_v1'
        assert audit['completion_sha256'] == sha(directory / 'completion.json')
        assert audit['protocol_sha256'] == sha(directory / 'protocol.json')
        assert status['audit_sha256'] == sha(directory / 'plotting_audit.json')
        for filename, digest in done['metadata_sha256'].items():
            assert sha(directory / filename) == digest
        assert done['n'] == 81 and done['arms'] == 55
        assert protocol['included_ids'] == frozen['included_ids']
        parts = []
        for filename, digest in sorted(done['files'].items()):
            path = directory / filename
            assert sha(path) == digest
            parts.append(torch.load(path, weights_only=True))
        ids = torch.cat([p['ids'] for p in parts]).tolist()
        assert ids == frozen['included_ids']
        loss_parts = []
        for part in parts:
            logits = part['initial']['decoded_logits'][:, :, 1:, 1:10]
            targets = part['y'][None, :, 1:] - 1
            targets = targets.expand(logits.shape[:-1])
            ce = -logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1).mean(-1)
            mse = (logits.softmax(-1) - torch.nn.functional.one_hot(targets, 9)).square().mean((-2, -1))
            loss_parts.append(torch.stack([part['initial']['error'], ce, mse], -1))
        losses = torch.cat(loss_parts, 1).double().numpy()
        score = torch.cat([p['initial']['critic'] for p in parts], 1).double().numpy()
        exact = torch.cat([p['exact'] for p in parts], 1).numpy()
        column = NAMES.index(name)
        arms = protocol['arms']
        rows = []
        for q in frozen['signed_state_displacements']:
            ai = next((i for i, arm in enumerate(arms) if arm['kind'] == 'critic' and arm['signed_state_displacement'] == q))
            cc = [i for i, arm in enumerate(arms) if arm['kind'] == 'control' and arm['signed_state_displacement'] == q]
            assert len(cc) == 8
            raw = losses[ai, :, column] - losses[0, :, column]
            corrected = losses[ai, :, column] - losses[cc, :, column].mean(0)
            rows.append(dict(signed_state_displacement=q, mean_probe_change=float((score[ai] - score[0]).mean()), mean_actual_target_change=float(raw.mean()), mean_actual_target_control_corrected_change=float(corrected.mean()), fraction_actual_target_change_matching_signed_direction=float((raw * np.sign(q) > 0).mean()), fraction_actual_target_unchanged=float((raw == 0).mean()), short_window_exact_count=int(exact[ai].sum()), native_short_window_exact_count=int(exact[0].sum())))
        np.savez_compressed(directory / 'actual_answer_errors.npz', ids=ids, error_names=NAMES, actual_errors=losses, predicted_critic=score)
        (directory / 'actual_answer_error_audit.json').write_text(json.dumps(dict(target=name, rows=rows, definition=frozen['targets'], scope='Posthoc readout diagnostic from saved logits; not an extra intervention or fitted model.', files={'actual_answer_errors.npz': sha(directory / 'actual_answer_errors.npz')}), indent=2) + '\n')
        probe_dir = Path(frozen['probe_sources'][name])
        metrics = [row for row in csv.DictReader((probe_dir / 'metrics.csv').open()) if row['split'] == 'test']
        figures = read(directory / 'intervention_statistics.json')
        paired = read(directory / 'paired_effect_tests.json')
        assert figures['target'] == paired['target'] == name
        assert figures['protocol'] == paired['protocol'] == 'fixed_state_displacement_v1'
        results[name] = dict(probe_test_metrics=metrics, intervention_statistics=figures, paired_tests=paired, actual_target_diagnostics=rows, audit=audit, completion=done)
    (OUT / 'RESULTS.json').write_text(json.dumps(dict(protocol=frozen, results=results), indent=2, allow_nan=False) + '\n')

    def metric(name, method, field='pearson_error'):
        return float(next((row for row in results[name]['probe_test_metrics'] if row['method'] == method))[field])
    lines = ['# Original E, answer CE and answer MSE: fixed-state-displacement interventions', '', 'All three groups use the same frozen HRM checkpoint, original 512-puzzle probe split and 81-puzzle intervention cohort. Network weights and puzzle selection are unchanged.', '', 'CE is nine-class cross-entropy averaged over all 81 cells, including clues and excluding BOS. MSE averages the 81×9 squared differences between those probabilities and one-hot answers. The extra H diagnostic readout is not written back to the state.', '', 'Original E retains the row/column/box excess-probability-mass measure and is a different supervision target from answer CE/MSE.', '', '## Main results and interpretation', '', 'Answer error is readable from L: on independent test puzzles, L-only linear-probe correlations are 0.8905/0.8877 for CE/MSE, and nonlinear-probe correlations are about 0.94. Correlations cannot be interpreted as equal fractions of explained information.', '', 'All three targets show the same next-step pattern: relative to matched random controls, directions that lower/raise probe scores decrease/increase the mean next-L step size. This does not imply the same direction for each of the 81 puzzles. Paired t-tests for all six doses per target meet the 0.05 threshold after within-target Holm correction; sign-flip sensitivity tests also pass. Effects are numerically similar across targets, but no equivalence test was performed.', '', 'A key limitation is that probe scores and actual answer errors do not track each other after intervention. Mean actual-target changes are positive for all three negative CE/MSE doses and negative for positive doses. This is a descriptive check of saved logits, without significance inference for this supplementary metric. Results support readable answer error and effects of the associated state direction on subsequent step size; they do not establish actual answer error as an independent causal step-control signal or improved solving accuracy.', '', '## Probe results (original 128 test puzzles)', '', '| Target | H-only r | L-only r | HL r | r for three MLP seeds |', '|---|---:|---:|---:|---|']
    for name in NAMES:
        mlp = ', '.join((f"{metric(name, f'mlp_seed{s}'):.4f}" for s in [1, 2, 3]))
        lines.append(f"| {LABEL[name]} | {metric(name, 'linear_H'):.4f} | {metric(name, 'linear_L'):.4f} | {metric(name, 'linear_HL'):.4f} | {mlp} |")
    lines += ['', '## Revised intervention protocol', '', 'The earlier relative-critic-dose protocol encountered negative OLS scores for 3/2 CE/MSE puzzles and unreachable ±30% doses for 2/7 puzzles. The revised protocol uses the 0.2%, 1% and 3% state displacements already supported by the original intervention code, in both increasing and decreasing critic directions. All 81 puzzles are retained; negative OLS predictions are neither clipped nor shifted.', '', 'Original E was rerun under the same revised protocol. The three groups have equal L-state displacement budgets at each dose, not equal critic-score changes. L is perturbed after H4 with H initially unchanged, followed by 12L+2H. Each dose has eight random controls matching per-token norm and displacement while preserving critic scores. Each group has 55 branches and 4,455 continuations.', '', 'Positive and negative doses with equal absolute displacement share the same random controls and do not increase the independent sample count. The statistical unit remains the 81 puzzles, with eight controls averaged within each puzzle.', '', '## Next-L step-size differences', '', 'The table reports intervention minus matched-control mean L2 step-size differences and pointwise 95% bootstrap intervals, using 10,000 paired whole-puzzle resamples over whole puzzles.', '', '| Signed state displacement | Original E | Answer CE | Answer MSE |', '|---|---:|---:|---:|']
    for q in frozen['signed_state_displacements']:
        cells = [f'{100 * q:+g}%']
        for name in NAMES:
            row = next((row for row in results[name]['intervention_statistics']['rows'] if row['intervention_percent'] == 100 * q))
            cells.append(f"{row['paired_control_corrected_mean']:+.4f} [{row['paired_control_corrected_ci95_low']:+.4f}, {row['paired_control_corrected_ci95_high']:+.4f}]")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Holm correction is applied separately to six doses per target, not to 18 pooled tests. Whether a pointwise interval crosses zero does not replace the corrected significance decision. Full t-tests and 100,000-draw sign-flip sensitivity results are in each target directory under PAIRED_TESTS_ZH.md.', '', '## Proxy scores and actual errors', '', 'Computed from extra H diagnostic logits saved after injection and before subsequent L/H updates, using the original answers. Increasing a probe score does not ensure increased actual error for every puzzle. This is supplementary description, not new training or post hoc selection. Error units differ across targets and their magnitudes are not directly comparable.', '', '| Target | State displacement | Mean probe-score change | Actual-target error change (control-corrected) | Fraction with raw actual-error change in the dose direction |', '|---|---:|---:|---:|---:|']
    for name in NAMES:
        for row in results[name]['actual_target_diagnostics']:
            if abs(row['signed_state_displacement']) == 0.03:
                lines.append(f"| {LABEL[name]} | {100 * row['signed_state_displacement']:+g}% | {row['mean_probe_change']:+.6g} | {row['mean_actual_target_control_corrected_change']:+.6g} | {100 * row['fraction_actual_target_change_matching_signed_direction']:.1f}% |")
    lines += ['', '## Scope and verification', '', 'Original data, earlier protocol failure records and figures are retained. Probe features and step deltas use the original frozen values. All available native-branch non-norm observations match earlier records bitwise; float32 norms allow the prespecified maximum difference of 4 ULP, with measured differences retained.', '', 'This follow-up uses one fixed checkpoint and an existing development cohort. Correlation is not a fraction of information, and a state-direction intervention does not prove causal mediation by an independent critic scalar. Continuations stop at H6 and cannot establish improved full-T16 solving accuracy.', '', f"[Machine-readable results]({OUT / 'RESULTS.json'}) · [Earlier protocol and blocking conditions]({ROOT / 'ORIGINAL_PROTOCOL_REPORT.md'})", '']
    for name in NAMES:
        lines += [f'## {LABEL[name]}: four figures', '']
        for stem in ['probe_correlations', 'l_full_sequence', 'l_early_sequence', 'l_intervention_step_size']:
            path = OUT / name / (stem + '.png')
            assert path.exists()
            readable = OUT / name / (stem + '_readable.png')
            if readable.exists():
                path = readable
            lines.append(f'- [{stem}]({path})')
        lines.append('')
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'report': str(OUT / 'REPORT.md'), 'targets': NAMES, 'status': 'complete'}))
if __name__ == '__main__':
    main()
