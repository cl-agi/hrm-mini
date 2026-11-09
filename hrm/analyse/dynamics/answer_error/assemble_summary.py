"""Assemble a Chinese summary solely from completed, preserved results."""
from hrm.paths import analysis_directory
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = analysis_directory(__file__)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    corr_path = ROOT / 'results.json'
    h_path = ROOT / 'fixed_state_H/H_INTERVENTION_RESULTS.json'
    l_path = ROOT.parent / 'fixed_state/RESULTS.json'
    audit_path = ROOT / 'independent_H_endpoint_audit.json'
    corr, h, old = read(corr_path), read(h_path), read(l_path)
    assert corr['status'] == h['status'] == 'complete'
    assert read(audit_path)['status'] == 'passed'
    assert h['correlation_results_sha256'] == sha(corr_path)
    assert h['report_script_sha256'] == sha(ROOT / 'report_h_intervention.py')
    names = {'original_excess': 'Constraint error E', 'ce': 'Answer CE', 'mse': 'Answer MSE'}
    targets = list(names)
    lines = [
        '# H-only interventions and correlations among three error measures', '',
        'Three H-only intervention sets are complete: 81 puzzles and 55 branches per set, totaling 13,365 short continuations. The model, checkpoint, inputs, answers, original 81 puzzles and H4 intervention point are unchanged. No model retraining, probe refitting or puzzle reselection was performed. The original L experiment and earlier results are retained.', '',
        'All three error measures can be read out from H, but decreasing the H-probe score increases the next L step size and increasing it decreases the step size, opposite to the earlier L-direction interventions. Strong CE/MSE correlations along native trajectories do not ensure faithful probe scores after intervention. These results do not support a shared higher-critic/larger-step interpretation across H and L.', '',
        '## H-only error readout', '',
        'Results use the original 128 independent test puzzles, with 192 pre-L states per puzzle. The 95% intervals use 10,000 whole-puzzle bootstrap samples with fixed probes and do not include model/probe retraining variability.', '',
        '| Target | H-only Pearson r [95% CI] | r after centering at each time point | L-only r (reference) |',
        '|---|---:|---:|---:|',
    ]
    for row in corr['cohorts']['test']['h_probes']:
        name = names['original_excess' if row['target'] == 'error' else row['target']]
        lo, hi = row['pearson_ci95']
        lines.append(f"| {name} | {row['pearson']:.4f} [{lo:.4f}, {hi:.4f}] | {row['pearson_time_centered']:.4f} | {row['reference_L_pearson']:.4f} |")
    lines += ['',
        'H changes only every six L updates, whereas diagnostic error can change at each L update; 192 time points correspond to 32 H phases. Lower raw H correlation does not directly imply missing error information. Time centering subtracts the across-puzzle mean at each time point, preserving within-time differences; it is neither causal adjustment nor an average of per-time correlations.', '',
        '## H intervention: next L update (prespecified primary endpoint)', '',
        'Dose is signed relative H-state displacement, not percentage change in probe score. A negative sign denotes the direction that lowers the H-only probe score. Eight random controls per puzzle match displacement and per-token norms while preserving the probe score. Controls are averaged within each puzzle before statistics across 81 puzzles. The table reports mean L2 step-size differences: directional intervention minus matched controls.', '',
        '| Relative H-state displacement | E | CE | MSE |', '|---|---:|---:|---:|']
    for j, q in enumerate(h['targets']['ce']['doses']):
        cells = [f'{100*q:+g}%']
        for target in targets:
            row = h['targets'][target]['endpoints']['primary_next_L']['rows'][j]
            assert row['dose'] == q
            value = row['paired_control_corrected']
            cells.append(f"{value['mean']:+.4f} [{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}]")
        lines.append('| ' + ' | '.join(cells) + ' |')
    max_p = max(row['p_t_holm'] for result in h['targets'].values()
                for row in result['endpoints']['primary_next_L']['rows'])
    lines += ['',
        f'All six doses for each target pass two-sided paired t-tests with within-target six-dose Holm correction; the largest adjusted p is {max_p:.6g}; all 100,000-draw sign-flip sensitivity tests also pass. Targets are not pooled into 18 tests. Intervals are pointwise 95% intervals; conclusions concern across-puzzle mean effects, not identical directions for every puzzle. The mean unperturbed next-L step size is 59.9474.', '',
        'Comparison with control-corrected differences from the earlier L interventions at ±3%:', '',
        '| Target | L intervention −3% | L intervention +3% | H intervention −3% | H intervention +3% |', '|---|---:|---:|---:|---:|']
    for target in targets:
        lr = old['results'][target]['intervention_statistics']['rows']
        lr = {row['intervention_percent']: row for row in lr}
        hr = h['targets'][target]['endpoints']['primary_next_L']['rows']
        lines.append(f"| {names[target]} | {lr[-3.]['paired_control_corrected_mean']:+.4f} | {lr[3.]['paired_control_corrected_mean']:+.4f} | {hr[0]['paired_control_corrected']['mean']:+.4f} | {hr[-1]['paired_control_corrected']['mean']:+.4f} |")
    lines += ['',
        'H and L interventions act through different paths, so these values cannot rank modules by critic ability. For the next L step F_L(l+h+x)−l, an H perturbation changes the conditioning input, while an L perturbation also changes the step origin; linearized responses contain Jδh and (J−I)δl, respectively. Equal displacement does not imply equal critic change or the same causal intervention.', '',
        '## First subsequent H update (prespecified secondary endpoint)', '',
        'This H update follows six ordinary L updates and measures the total dynamic effect propagated through L, not an isolated direct H effect. The mean unperturbed step size is 49.3831.', '',
        '| Target | Step-size difference at −3% | Holm p | Step-size difference at +3% | Holm p |', '|---|---:|---:|---:|---:|']
    for target in targets:
        rows = h['targets'][target]['endpoints']['secondary_first_H']['rows']
        a, b = rows[0], rows[-1]
        lines.append(f"| {names[target]} | {a['paired_control_corrected']['mean']:+.4f} | {a['p_t_holm']:.6g} | {b['paired_control_corrected']['mean']:+.4f} | {b['p_t_holm']:.6g} |")
    lines += ['',
        'After six-dose correction, only CE/MSE at +3% meet the 0.05 threshold; no E dose does. Sign-flip sensitivity results agree. This does not imply significant changes for all H step sizes, and the secondary endpoint cannot replace the primary endpoint. See target-specific reports for all six-dose intervals and p values.', '',
        '## Do probe changes track actual error changes?', '',
        'All readouts below are taken after injection and before subsequent updates. Probe changes are relative to the unperturbed branch; actual-target changes subtract matched controls and use the extra H diagnostic readout lm_head(H_level(H+L)), without writing it back to the state. Intervals are supplementary descriptive pointwise intervals, with no additional multiple-testing conclusions.', '',
        '| Target | H displacement | Probe prediction change | Actual-target readout change [95% CI] |', '|---|---:|---:|---:|']
    for target in targets:
        result = h['targets'][target]
        for j in [0, 5]:
            p = result['initial_readouts']['critic']['rows'][j]
            a = result['initial_readouts'][result['actual_target_field']]['rows'][j]['paired_control_corrected']
            lines.append(f"| {names[target]} | {100*p['dose']:+g}% | {p['change_from_native']['mean']:+.6g} | {a['mean']:+.6g} [{a['ci95'][0]:+.6g}, {a['ci95'][1]:+.6g}] |")
    lines += ['',
        'Mean changes in actual CE/MSE error oppose the probe direction; both E intervals cross zero. Predicting error on native trajectories does not ensure accuracy after moving off those trajectories along a probe direction. Lower predicted error therefore does not mean lower actual error. An H pulse also immediately changes native lm_head(H) output, unlike an L pulse; this separate auxiliary readout is saved in JSON and must not be confused with the extra H diagnostic readout.', '',
        '## Correlations among CE, MSE and constraint error', '',
        '| Measured error pair | Pearson r | Spearman ρ | Pearson r after centering at each time point |', '|---|---:|---:|---:|']
    for row in corr['cohorts']['test']['error_pairs']:
        translate = lambda key: names['original_excess' if key == 'error' else key]
        lines.append(f"| {translate(row['x'])} / {translate(row['y'])} | {row['pearson']:.4f} | {row['spearman']:.4f} | {row['pearson_time_centered']:.4f} |")
    lines += ['',
        'CE/MSE are strongly correlated on these native trajectories and should not be treated as fully independent evidence. E is related but not equivalent: it measures excess probability mass in rows, columns and boxes. Uniform 1/9 predictions yield E = 0 while CE = ln9 and MSE = 8/81, so E is not answer accuracy. See the correlation report for full intervals and definitions.', '',
        '## Verification and scope', '',
        'All original 81 puzzles are retained, including 1/5/3 nonpositive baseline H-probe scores for E/CE/MSE. Checks passed for all geometric branches, actual CE/MSE denominators, independent H-only score reconstruction, unchanged H scores during both six-L-update intervals, and matched controls. Comparable native non-norm quantities match the historical records bitwise; step norms differ by at most 2 ULP, below the prespecified 4 ULP threshold. Full hidden states are absent from the archive, so no full-state bitwise replay is claimed. Independent recomputation of all 36 endpoint statistics agrees with the report.', '',
        'This is a mechanism follow-up using one fixed checkpoint and an existing development cohort, not causal confirmation on a new independent cohort. Each intervention continues only 12L+2H through H6; full T16 solving was not rerun. Step-size changes alone do not establish reasoning-quality changes. Results support readable error information and dynamic effects of associated state directions, but do not establish explicit control by an independent critic scalar.', '',
        '## Reports, figures and reproduction paths', '',
        f'- [Correlation report and definitions]({ROOT}/CORRELATIONS_ZH.md)',
        f'- [Correlations among the three errors]({ROOT}/true_error_correlations.png)',
        f'- [H-only probe figure]({ROOT}/h_only_correlations.png)',
        f'- [Complete H-intervention statistics]({h_path})',
        f'- [Independent audit]({audit_path})',
        f'- [Frozen H-intervention protocol]({ROOT}/h_intervention_protocol.json)',
        f'- [H-intervention code]({ROOT}/run_h_intervention.py)',
    ]
    for target in targets:
        lines.append(f'- [{names[target]}: full time course, first three L updates, next-L dose curve and first-H dose curve]({ROOT}/fixed_state_H/{target}/H_INTERVENTION_REPORT.md)')
    out = ROOT / 'REPORT.md'
    with out.open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    sources = [corr_path, h_path, l_path, audit_path, Path(__file__).resolve().parent / 'assemble_summary.py']
    receipt = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
                   source_sha256={str(path): sha(path) for path in sources}, report_sha256=sha(out))
    with (ROOT / 'summary_receipt.json').open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(out)


if __name__ == '__main__':
    main()
