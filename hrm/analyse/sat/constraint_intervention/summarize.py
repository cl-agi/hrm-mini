"""Summarize the completed frozen run; never modify run artifacts or code."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np

from . import study
from hrm.paths import work_path
OUT = study.DIRECTORY/'summary'
OUT.mkdir(parents=True,exist_ok=True)
ROOT = study.ROOT
RUN = study.RUN


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def interval(lo, hi, digits=3, signed=False):
    spec = ('+' if signed else '') + f'.{digits}f'
    return f'[{format(lo, spec)}, {format(hi, spec)}]'


def main():
    status = read(RUN / 'status.json')
    assert status['status'] == 'COMPLETE'
    protocol = read(study.PROTOCOL)
    n_test=protocol['data']['test']
    fit = read(RUN / 'probes' / 'fit_summary.json')
    intervention = read(RUN / 'reports' / 'intervention_summary.json')
    manifest_hash = digest(RUN / 'run_manifest.json')
    assert all(d['run_manifest_sha256'] == manifest_hash for d in (status, fit, intervention))
    with np.load(RUN / 'reports' / 'per_puzzle_effects.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    assert arrays['critic'].shape == (12, n_test, 5)
    assert np.array_equal(arrays['case_id'], np.arange(n_test))
    assert np.allclose(arrays['effect'], arrays['critic'] - arrays['control_mean'], atol=0, rtol=0)

    probe_rows = []
    for module in ('H', 'L'):
        p = fit['probes'][module]
        row = {'module': module, 'alpha_selected_on_valid': p['alpha'], 'n_test_puzzles': n_test,
               'n_test_states': p['test']['estimate']['n']}
        for metric in ('pearson', 'r2', 'rmse'):
            row.update({metric: p['test']['estimate'][metric], metric + '_ci95_low': p['test']['ci'][metric][0],
                        metric + '_ci95_high': p['test']['ci'][metric][1]})
        probe_rows.append(row)
    native_rows = []
    for split, values in fit['native_success'].items():
        native_rows.append({'split': split, 'n_puzzles': values['n_puzzles'],
                            'success_any_outer_pct': 100 * values['success_any_outer_rate'],
                            'success_final_outer_pct': 100 * values['success_final_outer_rate']})

    primary_rows, outcome_rows = [], []
    for k, group in enumerate(intervention['comparisons']):
        module, dose = group['module'], group['signed_dose']
        assert arrays['module'][k] == module and arrays['signed_dose'][k] == dose
        treatment = arrays['critic'][k].mean(0)
        control = arrays['control_mean'][k].mean(0)
        native = arrays['native'].mean(0)
        primary = group['primary']
        assert np.isclose(primary['mean_effect'], arrays['effect'][k, :, 0].mean(), rtol=1e-12, atol=1e-12)
        changes = []
        name = 'critic_plus' if dose > 0 else 'critic_minus'
        for case in range(n_test):
            p = RUN / 'interventions' / 'test' / f'case_{case:04d}' / module / f'alpha_{abs(dose):g}' / f'{name}.json'
            changes.append(read(p)['initial_probe_change'])
        primary_rows.append({
            'module': module, 'signed_state_displacement_pct': 100 * dose,
            'probe_change_mean_predicted_clauses': float(np.mean(changes)),
            'critic_next_L_l2_mean': treatment[0], 'matched_control_next_L_l2_mean': control[0],
            'native_next_L_l2_mean': native[0], 'paired_effect_L2': primary['mean_effect'],
            'ci95_low': primary['ci'][0], 'ci95_high': primary['ci'][1],
            'effect_pct_of_control_mean': 100 * primary['mean_effect'] / control[0],
            'holm_p': primary['holm_pvalue'], 'n_puzzles': n_test,
        })
        for j, endpoint in enumerate(arrays['endpoints'][1:], 1):
            secondary = group['secondary'][endpoint]
            is_success = str(endpoint).startswith('success_')
            scale = 100 if is_success else 1
            outcome_rows.append({
                'module': module, 'signed_state_displacement_pct': 100 * dose, 'endpoint': endpoint,
                'mean_unit': 'percent' if is_success else 'clauses',
                'effect_unit': 'percentage_points' if is_success else 'clauses',
                'critic_mean': scale * treatment[j], 'matched_control_mean': scale * control[j],
                'native_mean': scale * native[j], 'paired_effect': scale * secondary['mean_effect'],
                'ci95_low': scale * secondary['ci'][0], 'ci95_high': scale * secondary['ci'][1],
                'inference': 'descriptive_puzzle_bootstrap_no_secondary_multiplicity_correction',
            })
    write_csv('probe_metrics.csv', probe_rows)
    write_csv('native_performance.csv', native_rows)
    write_csv('intervention_primary.csv', primary_rows)
    write_csv('intervention_outcomes.csv', outcome_rows)

    started = datetime.fromisoformat(read(RUN / 'run_manifest.json')['created_utc'])
    finished = datetime.fromisoformat(status['utc'])
    elapsed_minutes = (finished - started).total_seconds() / 60
    summary = {'run': str(RUN), 'run_manifest_sha256': manifest_hash,
               'started_utc': started.astimezone(timezone.utc).isoformat(),
               'finished_utc': finished.astimezone(timezone.utc).isoformat(), 'elapsed_minutes': elapsed_minutes,
               'native': native_rows, 'probe': probe_rows, 'primary': primary_rows, 'outcomes': outcome_rows}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')

    probe_table = table(['Probe', 'Pearson r (95% CI)', 'R² (95% CI)', 'RMSE: clause count (95% CI)'], [
        [r['module'], f"{r['pearson']:.4f} {interval(r['pearson_ci95_low'],r['pearson_ci95_high'],4)}",
         f"{r['r2']:.4f} {interval(r['r2_ci95_low'],r['r2_ci95_high'],4)}",
         f"{r['rmse']:.3f} {interval(r['rmse_ci95_low'],r['rmse_ci95_high'])}"] for r in probe_rows])
    native_table = table(['Dataset', 'Instances', f"Solved at least once within {protocol['trajectory']['outer_cycles']} inference cycles", f"Solved at inference cycle {protocol['trajectory']['outer_cycles']}"], [
        [r['split'], r['n_puzzles'], f"{r['success_any_outer_pct']:.2f}%", f"{r['success_final_outer_pct']:.2f}%"] for r in native_rows])
    primary_table = table(['Module', 'Signed state displacement', 'Mean probe change: clauses', 'Next-L displacement difference (95% CI)', 'Relative to control mean', 'Holm p'], [
        [r['module'], f"{r['signed_state_displacement_pct']:+g}%", f"{r['probe_change_mean_predicted_clauses']:+.3f}",
         f"{r['paired_effect_L2']:+.4f} {interval(r['ci95_low'],r['ci95_high'],4,True)}",
         f"{r['effect_pct_of_control_mean']:+.3f}%", f"{r['holm_p']:.3e}"] for r in primary_rows])
    final_table = table(['Module', 'State displacement', 'Mean final violated clauses: intervention', 'Random-control mean', 'Paired difference (95% CI)'], [
        [r['module'], f"{r['signed_state_displacement_pct']:+g}%", f"{r['critic_mean']:.3f}", f"{r['matched_control_mean']:.3f}",
         f"{r['paired_effect']:+.3f} {interval(r['ci95_low'],r['ci95_high'],3,True)}"]
        for r in outcome_rows if r['endpoint'] == 'final_violated_clauses'])
    success_any = [r for r in outcome_rows if r['endpoint'] == 'success_any_full_trajectory']
    success_final = [r for r in outcome_rows if r['endpoint'] == 'success_final_outer']
    success_table = table(['Module', 'State displacement', 'Within-budget success: intervention/control', 'Difference in pp (95% CI)', 'Final success: intervention/control', 'Difference in pp (95% CI)'], [
        [a['module'], f"{a['signed_state_displacement_pct']:+g}%", f"{a['critic_mean']:.2f}% / {a['matched_control_mean']:.2f}%",
         f"{a['paired_effect']:+.2f} {interval(a['ci95_low'],a['ci95_high'],2,True)}",
         f"{b['critic_mean']:.2f}% / {b['matched_control_mean']:.2f}%",
         f"{b['paired_effect']:+.2f} {interval(b['ci95_low'],b['ci95_high'],2,True)}"]
        for a, b in zip(success_any, success_final)])
    audit_path = OUT / 'integrity_audit.json'
    audit_text = 'The integrity audit is being summarized separately.'
    if audit_path.exists():
        audit = read(audit_path)
        assert audit['passed'] is True and audit['run_manifest_sha256'] == manifest_hash
        assert audit['verified_counts'] == audit['expected_counts']
        e = audit['maximum_actual_errors']
        audit_text = (f"Integrity checks passed for the supplied files and provenance; verified counts: {audit['verified_counts']}. "
                      f"Maximum relative per-token norm error {e['token_norm_relative_error']:.3e}, "
                      f"maximum absolute relative-displacement error {e['token_chord_absolute_error']:.3e}, "
                      f"maximum pooled random-control probe-score drift {e['control_pooled_probe_absolute']:.3e} predicted clauses. "
                      "All satisfy the original thresholds; see [integrity_audit.json](integrity_audit.json).")
    text = f'''# SAT constraint-readout and intervention statistics

Generated at {finished.isoformat()}. Results describe the supplied records.

## Integrity checks

{audit_text}

## Readout

{probe_table}

## Native performance

{native_table}

## Change in the next L update

{primary_table}

## Final violated clauses

{final_table}

## Solving outcomes

{success_table}

Random-control rates average control branches; paired statistics use instances as the unit.
See `summary.json` and the generated CSV tables for numerical details and uncertainty.
'''
    (OUT / 'RESULTS_SUMMARY.md').write_text(text)
    sources = [study.PROTOCOL, RUN / 'run_manifest.json', RUN / 'status.json',
               RUN / 'probes' / 'fit_summary.json', RUN / 'reports' / 'intervention_summary.json',
               RUN / 'reports' / 'per_puzzle_effects.npz']
    if audit_path.exists():
        sources.append(audit_path)
    (OUT / 'source_manifest.json').write_text(json.dumps({str(p): digest(p) for p in sources}, indent=2) + '\n')
    print(json.dumps({'summary': str(OUT / 'RESULTS_SUMMARY.md'), 'probe_rows': len(probe_rows),
                      'primary_rows': len(primary_rows), 'outcome_rows': len(outcome_rows),
                      'elapsed_minutes': elapsed_minutes}))


if __name__ == '__main__':
    main()
