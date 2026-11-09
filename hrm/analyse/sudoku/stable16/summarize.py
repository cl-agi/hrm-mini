"""Audit saved stable16 curves and report separate across-run mean/sample SD."""
import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

from hrm.paths import repository_root
from .evaluate_distributed import load_result_shards, validate_complete_results, atomic_json


def read(path):
    return json.loads(path.read_text())


def audit_run(directory):
    plan = read(directory/'run_config.json')
    combined = read(directory/'summary.json')
    budgets = plan['cycles']
    if len(budgets) < 2 or budgets != sorted(set(budgets)) or 16 not in budgets:
        raise ValueError('E10 curve requires multiple increasing budgets including 16')
    if plan['stable_cycles'] != 16:
        raise ValueError('E10 requires stable_cycles=16')
    if not combined.get('complete') or combined.get('cycle_budgets') != budgets or not combined.get('shared_inference'):
        raise ValueError('Incomplete or incompatible combined curve')
    if combined.get('checkpoint') != plan['checkpoint'] or combined.get('stable_cycles') != 16:
        raise ValueError('Combined curve checkpoint/stopping rule differs from the frozen plan')
    if set(combined['results']) != {str(c) for c in budgets}:
        raise ValueError('Combined curve is missing or has extra budgets')
    results = {}
    for limit in budgets:
        folder = directory/f'cycles_{limit}'
        expected_shards = {f'rank_{rank:05d}.npz' for rank in range(plan['world_size'])}
        actual_shards = {p.name for p in folder.glob('rank_*.npz')}
        if actual_shards - expected_shards:
            raise ValueError('Unexpected result shard outside the declared worker set')
        summary = read(folder/'summary.json')
        if (not summary.get('complete') or summary.get('input_provenance') != plan or
                summary.get('checkpoint') != plan['checkpoint'] or summary.get('stable_cycles') != 16 or
                summary.get('world_size') != plan['world_size'] or summary.get('requested_max_cycles') != limit or
                summary.get('shared_inference_max_cycles') != budgets[-1]):
            raise ValueError(f'Budget {limit} does not match the frozen run')
        measured = validate_complete_results(load_result_shards(folder, plan['world_size'], 0), plan['samples'], limit, 16)
        entry = combined['results'][str(limit)]
        if entry.get('summary') != f'cycles_{limit}/summary.json':
            raise ValueError('Unexpected budget summary path')
        if any(summary.get(k) != v or entry.get(k) != v for k,v in measured.items()):
            raise ValueError(f'Budget {limit} summary differs from actual shards')
        results[str(limit)] = measured
    return plan, results


def summarize(manifest, expected_runs=3, require_selection=False, require_e10_training=False):
    manifest = Path(manifest).resolve()
    specification = read(manifest)
    entries = specification['runs']
    if not entries: raise ValueError('No runs declared')
    groups = {}; identities=set(); checkpoints=set(); cohort=None; budgets=None; training_cohort=None
    for entry in entries:
        family, seed = entry['family'], entry['training_seed']
        if family not in ('HRM','RT') or type(seed) is not int:
            raise ValueError('Declare family HRM/RT and an integer training_seed')
        if (family,seed) in identities: raise ValueError('Repeated training seed within a family')
        identities.add((family,seed))
        directory = Path(entry['directory'])
        if not directory.is_absolute(): directory=manifest.parent/directory
        directory=directory.resolve()
        plan, result = audit_run(directory)
        arch = plan['config']['arch']['name']
        expected = {'HRM':{'hrm@HRM','hrm.models.hrm@HRM'},'RT':{'rt@RecurrentTransformer','hrm.models.rt@RecurrentTransformer'}}
        if arch not in expected[family]: raise ValueError('Declared family differs from model configuration')
        if any(key in plan['config']['arch'] for key in ('dual_module','hh_link','sym_io','readout','bp_steps')):
            raise ValueError('E10 aggregation requires the standard architecture, not an ablation')
        if plan['config'].get('transformer',{}).get('is_mlp_mixer') or plan['config']['arch'].get('is_mlp_mixer'):
            raise ValueError('E10 HRM/RT aggregation does not mix Mixer curves')
        digest=plan['checkpoint_sha256']
        if digest in checkpoints: raise ValueError('One checkpoint cannot stand for multiple training runs')
        checkpoints.add(digest)
        current=(plan['samples'],plan['dataset_order_sha256'],plan['split'])
        if cohort is not None and current != cohort: raise ValueError('All runs must evaluate the same ordered cohort')
        if budgets is not None and budgets != plan['cycles']: raise ValueError('Runs have different cycle budgets')
        cohort=current;budgets=plan['cycles']
        selection = None
        if entry.get('selection_receipt'):
            from hrm.training.records import verify_best_selection
            receipt = Path(entry['selection_receipt'])
            if not receipt.is_absolute(): receipt=manifest.parent/receipt
            selection=verify_best_selection(receipt,checkpoint_sha256=digest,seed=seed,cycles=16,weights='EMA')
        elif require_selection:
            raise ValueError('Missing 16-cycle EMA checkpoint-selection receipt')
        training = None
        if entry.get('training_directory'):
            from hrm.training.data_scaling_records import verify_e10_training
            training_directory=Path(entry['training_directory'])
            if not training_directory.is_absolute():training_directory=manifest.parent/training_directory
            training=verify_e10_training(training_directory,plan,seed)
            training['fixed16_reevaluation_correct_delta']=result['16']['correct']-training['selection_correct']
            if training_cohort is not None and training['train']!=training_cohort:
                raise ValueError('E10 runs must share the same training draw')
            training_cohort=training['train']
        elif require_e10_training:
            raise ValueError('Missing native E10 training directory')
        groups.setdefault(family,[]).append(dict(training_seed=seed,directory=str(directory),checkpoint_sha256=digest,
            scalar_selection_verified=selection is not None,selection=selection,
            e10_training_verified=training is not None,training=training,results=result))
    output={}
    for family,runs in groups.items():
        if len(runs) != expected_runs: raise ValueError(f'{family}: expected {expected_runs} runs, found {len(runs)}')
        curve=[]
        for c in budgets:
            accuracies=[100*r['results'][str(c)]['correct']/cohort[0] for r in runs]
            mean_cycles=[r['results'][str(c)]['mean_cycles'] for r in runs]
            curve.append(dict(inference_cycles=c,n_runs=len(runs),accuracy_percent_mean=statistics.mean(accuracies),
                accuracy_percent_sample_sd=statistics.stdev(accuracies) if len(runs)>1 else None,
                mean_actual_cycles=statistics.mean(mean_cycles)))
        output[family]=dict(runs=runs,curve=curve)
    return dict(groups=output,samples=cohort[0],dataset_order_sha256=cohort[1],split=cohort[2],
        stable_cycles=16,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        interpretation='Across-run sample SD (ddof=1), not SEM or a confidence interval. Each curve uses one frozen checkpoint across budgets.',
        limitations='Independent execution is not certified. Native training checks validate the supplied complete schedule, cohort and latest-tie weight identity; they cannot prove unrecorded execution. Completed budgets without native evidence are not certified. Optional selection receipts bind the evaluated weight hash to the earliest maximum fixed-16-cycle EMA scalar record in the supplied training log; they do not certify unrecorded training, data identity or missing evaluations.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--expected-runs',type=int,default=3)
    p.add_argument('--output',type=Path)
    p.add_argument('--require-selection',action='store_true',help='Require matching fixed-16-cycle EMA selection receipts for every run')
    p.add_argument('--require-e10-training',action='store_true',help='Require native fixed-update E10 schedule, data and latest-tie selection evidence')
    a=p.parse_args(argv)
    if a.expected_runs<1:p.error('--expected-runs must be positive')
    root=Path(os.environ.get('HRM_ANALYSIS_RECORDS',repository_root()/'exp/sudoku/records/stable_stopping'))
    manifest=a.manifest or root/'curve_runs.json'
    result=summarize(manifest,a.expected_runs,a.require_selection,a.require_e10_training)
    output=a.output or manifest.parent/'curve_summary.json'
    atomic_json(output,result)
    print(json.dumps(result))


if __name__=='__main__':main()
