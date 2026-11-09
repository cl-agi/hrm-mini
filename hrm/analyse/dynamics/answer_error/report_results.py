from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
"""Summarize the original and both answer-error experiments from saved results."""
from hrm.paths import repo_path, work_path
from pathlib import Path
from pathlib import Path
import csv
import hashlib
import json
ROOT = analysis_directory(__file__)
HEX = Path(str(work_path('cl_agi_hrm_mini/artifacts')))
TARGETS = [('original_excess', Path(str(repo_path('checkpoint/dynamics/cl_agi_hrm_mini/artifacts/internal_critic'))), HEX / 'critic_Lonly_20261004'), ('ce', ROOT / 'probes/ce', ROOT / 'interventions/ce'), ('mse', ROOT / 'probes/mse', ROOT / 'interventions/mse')]

def read(path):
    return json.loads(path.read_text())

def verify_files(directory, receipt):
    for name, expected in receipt['files'].items():
        with (directory / name).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    results = {}
    for name, probes, intervention in TARGETS:
        if not (probes / 'completion.json').exists() and name != 'original_excess':
            raise RuntimeError(f'{name}: probe fitting is incomplete')
        if name != 'original_excess':
            receipt = read(probes / 'completion.json')
            assert receipt['status'] == 'complete' and receipt['target'] == name
            verify_files(probes, receipt)
        with (probes / 'metrics.csv').open() as stream:
            metrics = [row for row in csv.DictReader(stream) if row['split'] == 'test']
        item = dict(probe_directory=str(probes), intervention_directory=str(intervention), probe_test_metrics=metrics, probe_results=read(probes / 'results.json'))
        if (intervention / 'precheck.json').exists():
            item['precheck'] = read(intervention / 'precheck.json')
        fully_complete = name == 'original_excess'
        if name != 'original_excess':
            needed = ['completion.json', 'wrapper_completion.json', 'plotting_status.json', 'plotting_audit.json']
            if all(((intervention / file).exists() for file in needed)):
                fully_complete = read(intervention / 'completion.json')['status'] == 'complete' and read(intervention / 'wrapper_completion.json')['status'] == 'complete' and (read(intervention / 'plotting_status.json')['status'] == 'completed') and (read(intervention / 'plotting_audit.json')['passed'] is True)
        if fully_complete:
            receipt = read(intervention / 'completion.json')
            assert receipt['status'] == 'complete' and receipt['n'] == 81 and (receipt['arms'] == 55)
            verify_files(intervention, receipt)
            item['status'] = 'complete'
            item['intervention_statistics'] = read(intervention / 'intervention_statistics.json')
            item['response'] = read(intervention / 'L_response_summary.json')
            item['paired_tests'] = read(intervention / 'paired_effect_tests.json')
            if name != 'original_excess':
                item['audit'] = read(intervention / 'plotting_audit.json')
                assert read(intervention / 'wrapper_completion.json')['criterion'] == name
                assert read(intervention / 'plotting_status.json')['target'] == name
                assert item['audit']['target'] == item['intervention_statistics']['target'] == item['paired_tests']['target'] == name
                assert item['audit']['new_completion_sha256'] == sha(intervention / 'completion.json')
                assert item['audit']['new_protocol_sha256'] == sha(intervention / 'protocol.json')
        else:
            item['status'] = 'intervention_not_completed'
            if (intervention / 'wrapper_status.json').exists():
                item['intervention_status'] = read(intervention / 'wrapper_status.json')
        results[name] = item
    output = dict(scope='Frozen HRM seed1, original512 probe subjects split256/128/128, original81 intervention subjects.', protocol=read(ROOT / 'protocol.json'), collection=read(ROOT / 'collection/manifest.json'), results=results)
    (ROOT / 'RESULTS.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    labels = {'original_excess': 'Original constraint-mass error E', 'ce': 'Answer CE', 'mse': 'Answer-probability MSE'}

    def metric(name, method, field):
        return float(next((row for row in results[name]['probe_test_metrics'] if row['method'] == method))[field])
    lines = ['# Answer CE/MSE critic experiments under the original conditions', '', 'Only the error-probe supervision target changes; the original HRM checkpoint remains frozen, with no HRM retraining or fine-tuning.', '', 'CE is nine-class cross-entropy for digits 1–9, averaged over all 81 cells. MSE averages the 81×9 squared differences between the same probabilities and one-hot answers. Both include clues and exclude BOS and class 0. Readout uses one extra H update that is not written back to the network state.', '', '## Probes: original 128 independent test puzzles', '', 'The original 512 puzzles and 256/128/128 split are unchanged, with 192 pre-L states per puzzle. OLS, three MLP seeds, training parameters, time-only controls and whole-puzzle label-shuffle controls follow the original implementation. The table reports Pearson r between predicted and target error.', '', '| Error definition | H-only r | L-only r | HL r | L-only R² | r between L-only prediction and next-L step size |', '|---|---:|---:|---:|---:|---:|']
    for name, _, _ in TARGETS:
        lines.append(f"| {labels[name]} | {metric(name, 'linear_H', 'pearson_error'):.6f} | {metric(name, 'linear_L', 'pearson_error'):.6f} | {metric(name, 'linear_HL', 'pearson_error'):.6f} | {metric(name, 'linear_L', 'r2_error'):.6f} | {metric(name, 'linear_L', 'pearson_next_L'):.6f} |")
    lines += ['', '| Error definition | MLP seed1 r | MLP seed2 r | MLP seed3 r | time-only r | Whole-puzzle shuffle r |', '|---|---:|---:|---:|---:|---:|']
    for name, _, _ in TARGETS:
        values = [metric(name, method, 'pearson_error') for method in ['mlp_seed1', 'mlp_seed2', 'mlp_seed3', 'time_only', 'puzzle_shuffled_linear']]
        lines.append('| ' + labels[name] + ' | ' + ' | '.join((f'{v:.6f}' for v in values)) + ' |')
    lines += ['', '## Intervention status for the fixed 81 puzzles', '', 'These 81 puzzles remain the development cohort selected by original joint E critic ≥ 1, without CE/MSE reselection. The original protocol requires positive new L-only baseline scores and reachable ±5%, ±15%, ±30% target doses within a 20% state-displacement limit.', '', '| Target | Intervention status | Minimum baseline score | Puzzles with nonpositive scores |', '|---|---|---:|---:|']
    for name in ['ce', 'mse']:
        item = results[name]
        check = item.get('precheck', {})
        lines.append(f"| {labels[name]} | {item['status']} | {check.get('score_min', 'Not recorded')} | {check.get('nonpositive_count', 'Not recorded')} |")
    for name in ['ce', 'mse']:
        item = results[name]
        if 'intervention_status' in item:
            lines += ['', f"{labels[name]}: {item['intervention_status']['error']}"]
    available = [name for name, _, _ in TARGETS if results[name]['status'] == 'complete']
    if len(available) > 1:
        lines += ['', '## Next L update: step-size differences after subtracting matched controls', '', 'Units are L2 norms of state updates. Eight score-preserving, displacement-matched random controls are averaged within each puzzle before paired differences; 95% intervals use the original 10,000 whole-puzzle bootstrap samples.', '', '| Target critic-score change | ' + ' | '.join((labels[n] for n in available)) + ' |', '|---|' + '---:|' * len(available)]
        for percent in [-30, -15, -5, 5, 15, 30]:
            row = [f'{percent:+d}%']
            for name in available:
                r = next((r for r in results[name]['intervention_statistics']['rows'] if r['intervention_percent'] == percent))
                row.append(f"{r['paired_control_corrected_mean']:+.4f} [{r['paired_control_corrected_ci95_low']:+.4f}, {r['paired_control_corrected_ci95_high']:+.4f}]")
            lines.append('| ' + ' | '.join(row) + ' |')
        lines += ['', 'The 95% intervals are pointwise. Six-dose Holm correction remains separate for each target, not pooled into 12 CE/MSE tests. Interval crossing does not replace the corrected significance decision. See PAIRED_TESTS_ZH.md in each directory for full paired t-tests and 100,000-draw sign-flip sensitivity tests.', '', 'Equal percentage changes in critic score need not require equal state displacements across error definitions. Median displacements are in L_response_summary.json; cross-target absolute effects should not be interpreted directly as stronger mechanisms.']
    lines += ['', '## Consistency and scope', '', 'Original puzzles, answers, IDs, checkpoint, covariance and intervention plan are unchanged. The plan specifies 55 branches and 12L+2H continuations; intervention_not_completed means that precheck stopped the group before continuation. New critic directions and their zero-projection controls change with new probe weights, directly reflecting the changed critic definition.', '', 'All probe fitting and correlation statistics use original frozen features and deltas. Replay requires bitwise agreement for saved non-norm observations; long-vector float32 norms are recomputation audits with a prespecified 4 ULP allowance. Original code, initial strict-check failures, diagnostics and revisions are retained. No original full hidden tensors are available, so no bitwise comparison of those tensors is claimed.', '', 'This error-readout and state-intervention study uses one fixed HRM checkpoint and an existing development cohort. Correlation is not a fraction of information; direction interventions alone do not establish effects mediated solely by one critic scalar. Continuations end at H6, not a full T16 solving evaluation. Unexecuted branches have no reported results.', '', '## Files', '', f"- [Machine-readable summary]({ROOT / 'RESULTS.json'})"]
    for name in ['ce', 'mse']:
        out = ROOT / 'interventions' / name
        lines += ['', f'### {name.upper()}', '']
        for stem in ['probe_correlations', 'l_full_sequence', 'l_early_sequence', 'l_intervention_step_size']:
            p = out / (stem + '.png')
            if p.exists():
                lines.append(f'- [{stem}]({p})')
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'report': str(ROOT / 'REPORT.md'), 'status': {k: v['status'] for k, v in results.items()}}))
if __name__ == '__main__':
    main()
