"""Export the frozen SAT top-up and retained main-task cohort. No network calls."""
from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
from pathlib import Path
import ast
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import mean, stdev
HERE = analysis_directory(__file__)
SOURCE_ROOT = Path(str(work_path('llm-solver-experiments')))
PROJECT = SOURCE_ROOT / '015_sat_topup_100'
INVOCATION = PROJECT / 'runs/seed_001/launches/20261014T090143_757343Z'
OLD = HERE.parent / 'llm_reference'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def read(path):
    return json.loads(Path(path).read_bytes())

def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def write_csv(name, rows):
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def write_jsonl_gz(name, rows):
    data = ''.join((canonical(row) + '\n' for row in rows)).encode()
    (HERE / name).write_bytes(gzip.compress(data, mtime=0))

def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['task'], row['model'], str(row['scale'])].append(row)
    output = []
    for (task, model, scale), cases in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], int(x[0][2]) if x[0][0] == 'sat' else x[0][2])):
        durations = [c['elapsed_seconds'] / 60 for c in cases]
        providers = Counter((c['provider_configuration_json'] for c in cases))
        output.append(dict(task=task, model=model, scale=scale, n_trials=len(cases), n_unique_problems=len({c['problem_sha256'] for c in cases}), n_correct=sum((c['correct'] for c in cases)), accuracy=sum((c['correct'] for c in cases)) / len(cases), retained_trials=sum((c['origin'] == 'retained' for c in cases)), new_trials=sum((c['origin'] == 'new' for c in cases)), completed_trials=sum((c['selected_model_status'] == 'completed' for c in cases)), length_limited_trials=sum((c['selected_model_status'] == 'length_limited' for c in cases)), no_final_content_trials=sum((c['selected_model_status'] == 'no_final_content' for c in cases)), selected_duration_count=len(durations), mean_minutes=mean(durations), sample_sd_minutes=stdev(durations), provider_groups_json=canonical([dict(configuration=json.loads(k), n_trials=v) for k, v in sorted(providers.items())]), source_stages_json=canonical(dict(sorted(Counter((c['source_stage'] for c in cases)).items())))))
    return output

def cost_summary(rows, field='reported_cost_usd'):
    known = [Decimal(str(r[field])) for r in rows if r[field] is not None]
    return dict(attempt_count=len(rows), reported_cost_attempts=len(known), missing_cost_attempts=len(rows) - len(known), sum_reported_cost_usd=str(sum(known, Decimal(0))), complete=len(known) == len(rows))

def main():
    (HERE / 'sources').mkdir(exist_ok=True)
    (HERE / 'sources/prompts').mkdir(exist_ok=True)
    copied = []

    def copy(source, destination, compress=False):
        data = source.read_bytes()
        output = HERE / destination
        output.write_bytes(gzip.compress(data, mtime=0) if compress else data)
        copied.append(dict(source_path=str(source), package_path=destination, source_sha256=sha(data), source_bytes=len(data), package_sha256=sha(output.read_bytes()), encoding='gzip' if compress else 'byte-identical'))
    copy(INVOCATION / 'combined_sat/merged_report.json', 'sources/merged_report.json.gz', True)
    copy(INVOCATION / 'audit/audit_report.json', 'sources/audit_report.json.gz', True)
    copy(INVOCATION / 'completion.json', 'sources/completion.json')
    for name in ['FULL_PLAN.json', 'MODEL_PROFILES.json', 'PROVIDER_CHANGES.json']:
        copy(PROJECT / name, 'sources/' + name)
    copy(PROJECT / 'TOPUP_MANIFEST.json', 'sources/TOPUP_MANIFEST.json.gz', True)
    copy(PROJECT / 'full_inputs/source_manifest.json', 'sources/input_source_manifest.json')
    for name in ['sat3.txt', 'sudoku.txt', 'maze.txt']:
        copy(PROJECT / 'prompts' / name, 'sources/prompts/' + name)
    definitions = []
    code_sources = []
    for path, names in [(SOURCE_ROOT / '000_original/openrouter.py', ['robust_json_parse']), (PROJECT / 'runtime/audit_full.py', ['independent_validate', 'regrade'])]:
        raw = path.read_bytes()
        tree = ast.parse(raw)
        found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in found} == set(names)
        definitions.extend((ast.get_source_segment(raw.decode(), n) for n in found))
        code_sources.append(dict(source_path=str(path), source_sha256=sha(raw), extracted_functions=names))
    (Path(__file__).resolve().parent / 'frozen_grading.py').write_text('"""Pure definitions copied from the frozen parser and independent audit."""\nimport json\nimport re\nfrom collections import deque\nFLAGS = ("correct", "json_parse_success", "strict_json_object", "answer_schema_valid", "strict_output_format")\n\n' + '\n\n'.join(definitions) + '\n')
    import hrm.analyse.baselines.llm_sat.frozen_grading as frozen_grading
    merged = read(INVOCATION / 'combined_sat/merged_report.json')
    audit = read(INVOCATION / 'audit/audit_report.json')
    old_bytes = gzip.decompress((OLD / 'sources/merged_report.json.gz').read_bytes())
    old = json.loads(old_bytes)
    assert merged['complete'] and (not merged['errors']) and (merged['audited_complete_cases'] == 2400)
    assert audit['complete'] and audit['audited_complete_cases'] == 1260
    retained_main = [c for c in old['cases'] if c['task'] in ('sudoku', 'maze') and c['stage'] == 'main_100']
    assert len(retained_main) == 800
    cases = merged['cases'] + [dict(c, origin='retained') for c in retained_main]
    assert len({c['job_id'] for c in cases}) == len(cases) == 3200
    audit_by_id = {c['job_id']: c for c in audit['cases']}
    old_by_id = {c['job_id']: c for c in old['cases']}
    records, answers, questions, artifacts = ([], [], [], [])
    for c in sorted(cases, key=lambda x: x['job_id']):
        result_path = Path(c['selected_result_path'])
        raw = result_path.read_bytes()
        r = json.loads(raw)
        source_case = audit_by_id[c['job_id']] if c['origin'] == 'new' else old_by_id[c['job_id']]
        expected = c.get('selected_result_sha256')
        if expected is None:
            assert sha(raw) in {a.get('result_sha256') for a in source_case['attempts']}
        else:
            assert sha(raw) == expected
        assert r['correct'] == c['correct'] and r['status'] == c['selected_model_status']
        assert r['status'] in ('completed', 'length_limited', 'no_final_content')
        assert r['terminal_finish_reason'] == ('length' if r['status'] == 'length_limited' else 'stop')
        seconds = r['elapsed_seconds']
        assert isinstance(seconds, (int, float)) and math.isfinite(seconds) and (seconds >= 0)
        problem_path = result_path.parent / 'problem.json'
        request_path = result_path.parent / 'request.json'
        problem_raw, request_raw = (problem_path.read_bytes(), request_path.read_bytes())
        problem, request = (json.loads(problem_raw), json.loads(request_raw))
        assert sha(canonical(problem).encode()) == c['problem_sha256'] == r['problem_sha256']
        if c.get('selected_request_sha256'):
            assert sha(request_raw) == c['selected_request_sha256']
        grading = frozen_grading.regrade(c['task'], problem, r.get('final_content', ''), frozen_grading.robust_json_parse)
        assert grading['correct'] == r['correct']
        assert grading.get('parsed_answer') == r.get('parsed_answer')
        usage = r.get('usage') or {}
        record = dict(job_id=c['job_id'], task=c['task'], model=c['model'], scale=c['scale'], source_stage=c['stage'], origin=c['origin'], case_id=c['case_id'], seed_id=1, problem_sha256=c['problem_sha256'], correct=r['correct'], strict_output_format=r.get('strict_output_format'), selected_model_status=r['status'], terminal_finish_reason=r['terminal_finish_reason'], selected_attempt=int(result_path.parent.name.split('_')[-1]), selected_logical_attempt=source_case.get('selected_logical_attempt', source_case.get('selected_attempt')), provider_configuration_json=canonical(request['provider']), observed_providers_json=canonical(r.get('observed_providers', [])), elapsed_seconds=seconds, elapsed_minutes=seconds / 60, started_utc=r.get('started_utc'), finished_utc=r.get('finished_utc'), prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'), reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'), total_tokens=usage.get('total_tokens'), selected_reported_cost_usd=usage.get('cost'), selected_result_path=str(result_path), selected_result_sha256=sha(raw), selected_request_path=str(request_path), selected_request_sha256=sha(request_raw))
        records.append(record)
        answers.append(dict(record, final_content=r.get('final_content'), parsed_answer=r.get('parsed_answer'), usage=usage, payload_sha256=r.get('payload_sha256'), request_call_settings=r.get('request_call_settings'), response_model=r.get('response_model'), provider=r.get('provider'), configuration_check=r.get('configuration_check')))
        questions.append(dict(job_id=c['job_id'], task=c['task'], model=c['model'], scale=c['scale'], source_stage=c['stage'], origin=c['origin'], problem=problem, problem_sha256=c['problem_sha256']))
        selected_audit = [a for a in source_case['attempts'] if a.get('result_sha256') == sha(raw)]
        assert len(selected_audit) == 1
        artifacts.append(dict(job_id=c['job_id'], selected_result_path=str(result_path), selected_result_sha256=sha(raw), selected_request_path=str(request_path), selected_request_sha256=sha(request_raw), selected_problem_path=str(problem_path), selected_problem_file_sha256=sha(problem_raw), problem_sha256=c['problem_sha256'], selected_raw_events_path=str(result_path.parent / 'events.jsonl'), selected_raw_events_sha256_from_audit=selected_audit[0].get('events_sha256'), events_hash_note='Hash copied from original independent audit; event bytes are not duplicated or rehashed by this exporter.', selected_attempt_provenance=source_case.get('selected_attempt_provenance')))
    settings = summarize(records)
    sat_settings = [s for s in settings if s['task'] == 'sat']
    assert len(settings) == 32 and len(sat_settings) == 24 and all((s['n_trials'] == 100 for s in settings))
    for s in sat_settings:
        source = next((x for x in merged['statistics'] if x['model'] == s['model'] and x['scale'] == int(s['scale'])))
        assert (source['correct'], source['unique_problem_hashes']) == (s['n_correct'], s['n_unique_problems'])
    write_csv('selected_cases.csv', records)
    timing_fields = ['job_id', 'task', 'model', 'scale', 'source_stage', 'origin', 'seed_id', 'problem_sha256', 'correct', 'selected_model_status', 'terminal_finish_reason', 'provider_configuration_json', 'observed_providers_json', 'elapsed_seconds', 'elapsed_minutes', 'started_utc', 'finished_utc', 'selected_result_path', 'selected_result_sha256']
    write_csv('timing.csv', [{k: r[k] for k in timing_fields} for r in records])
    write_csv('current_settings.csv', settings)
    write_csv('sat_accuracy.csv', sat_settings)
    write_csv('timing_summary.csv', settings)
    write_jsonl_gz('final_answers.jsonl.gz', answers)
    write_jsonl_gz('questions.jsonl.gz', questions)
    write_jsonl_gz('source_artifact_index.jsonl.gz', artifacts)
    attempts = []
    for c in sorted(audit['cases'], key=lambda x: x['job_id']):
        case_dir = PROJECT / 'runs/seed_001/cases' / c['job_id']
        summary = read(case_dir / 'case_summary.json')
        for a in summary['attempts']:
            path = Path(a['result_path'])
            raw = path.read_bytes()
            r = json.loads(raw)
            assert sha(raw) == a['result_sha256']
            independently_audited = next((x for x in c['attempts'] if x['attempt'] == a['attempt']))
            assert independently_audited['result_sha256'] == sha(raw)
            usage = r.get('usage') or {}
            attempts.append(dict(job_id=c['job_id'], model=c['model'], scale=c['scale'], attempt=a['attempt'], selected=a['attempt'] == summary['selected_attempt'], classification=a['classification']['kind'], classification_reason=a['classification']['reason'], status=r['status'], http_status=r.get('http_status'), error_codes_json=canonical(r.get('error_codes', [])), elapsed_seconds=r.get('elapsed_seconds'), reported_cost_usd=usage.get('cost'), prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'), total_tokens=usage.get('total_tokens'), usage_json=canonical(usage), result_path=str(path), result_sha256=sha(raw)))
    assert len(attempts) == 2545 and sum((a['selected'] for a in attempts)) == 1260
    write_csv('new_attempts.csv', attempts)
    selected_costs = [dict(reported_cost_usd=r['selected_reported_cost_usd']) for r in records]
    historical_costs = [dict(reported_cost_usd=r['selected_reported_cost_usd']) for r in records if r['origin'] == 'retained']
    accounting = dict(note='Timing summaries use only selected terminal requests. New attempt accounting includes all 015 attempts and engineering retries. Historical accounting covers only 1940 retained selected requests; historical retry charges are not reconstructed. Missing provider usage is unknown, never zero.', current_3200_selected_requests=cost_summary(selected_costs), historical_1940_selected_requests=cost_summary(historical_costs), new_015_all_attempts=dict(cost_summary(attempts), selected_trials=1260, additional_engineering_attempts=1285, classification_counts=dict(Counter((a['classification'] for a in attempts))), missing_usage_fields={field: sum((a[field] is None for a in attempts)) for field in ['reported_cost_usd', 'prompt_tokens', 'completion_tokens', 'total_tokens']}), new_015_by_model={model: dict(cost_summary(group), selected_trials=sum((a['selected'] for a in group)), engineering_attempts=sum((a['classification'] == 'retryable_transport' for a in group)), status_counts=dict(Counter((a['status'] for a in group)))) for model in sorted({a['model'] for a in attempts}) for group in [[a for a in attempts if a['model'] == model]]})
    assert accounting['new_015_all_attempts']['sum_reported_cost_usd'] == merged['costs']['new_all_attempts']['usage']['cost']['sum_reported']
    write_json('cost_accounting.json', accounting)
    write_json('sources/audit_summary.json', {k: v for k, v in audit.items() if k not in ['cases', 'statistics', 'comparison_contract']})
    excluded = [dict(job_id=c['job_id'], model=c['model'], task=c['task'], scale=c['scale'], source_stage=c['stage'], reason='Overlapping historical SAT sweep trial excluded by the pre-frozen top-up manifest') for c in old['cases'] if c['job_id'] not in {r['job_id'] for r in records}]
    assert len(excluded) == 100 and all((c['task'] == 'sat' for c in excluded))
    write_csv('excluded_historical_cases.csv', excluded)
    write_json('cohort_summary.json', dict(n_trials=len(records), n_settings=32, sat_trials=2400, retained_sat_trials=1140, new_sat_trials=1260, retained_sudoku_maze_trials=800, excluded_overlapping_historical_sat_trials=100, terminal_status_counts=dict(Counter((r['selected_model_status'] for r in records))), n_correct=sum((r['correct'] for r in records)), duration_count=len(records), duration_definition='Selected result elapsed_seconds / 60; client observed single-request elapsed duration, including service-side waiting, generation, streaming and network transfer. Local scheduler waiting and between-request retry backoff are outside this interval.', sd_definition='Sample standard deviation over the 100 selected request durations in each setting (ddof=1); not variability between independent seeds.', independent_api_call_groups=1, numeric_api_seed_sent=False, sat_unique_formulas_by_n={'8': 100, '16': 99, '32': 100, '64': 100, '128': 98, '256': 100}))
    write_json('source_manifest.json', dict(copied_sources=copied, extracted_code=code_sources, retained_archive=dict(path=str(OLD / 'sources/merged_report.json.gz'), package_relative_path='../llm_reference/sources/merged_report.json.gz', sha256=sha((OLD / 'sources/merged_report.json.gz').read_bytes()), uncompressed_sha256=sha(old_bytes)), raw_artifact_index='source_artifact_index.jsonl.gz', scope='3200 current selected trials = 2400 SAT from 015 + 800 archived Sudoku/Maze main_100. No thinking copied.'))
    write_report(settings)
    update_manifest()
    print(json.dumps(read(HERE / 'cohort_summary.json'), ensure_ascii=False, indent=2))

def write_report(settings):
    models = ['openai/gpt-5.2', 'google/gemini-3.1-pro-preview', 'anthropic/claude-opus-4.5', 'deepseek/deepseek-v3.2']
    groups = {(r['task'], r['model'], str(r['scale'])): r for r in settings}
    lines = ['# Results and request durations for the four selected models', '', 'The cohort contains 3,200 selected terminal model responses: 2,400 SAT trials from 015 (1,140 historical and 1,260 additional), plus 800 Sudoku/Maze trials from the reference cohort. There are 32 settings with 100 trials each. A frozen manifest selects the historical phase; 100 overlapping historical SAT sweep calls are excluded.', '', '## SAT accuracy', '', '| Variables | GPT-5.2 | Gemini 3.1 Pro Preview | Claude Opus 4.5 | DeepSeek V3.2 |', '|---:|---:|---:|---:|---:|']
    for n in [8, 16, 32, 64, 128, 256]:
        lines.append('| ' + str(n) + ' | ' + ' | '.join((str(groups['sat', m, str(n)]['n_correct']) + '/100' for m in models)) + ' |')
    lines.extend(['', 'The 100 trials at n16 contain 99 distinct formulas, and n128 contains 98; all other sizes contain 100. Repeated instances are retained. The original seed001 call group was completed to its target size; API requests did not specify a numerical seed.', '', '## Request duration: mean ± sample SD (minutes)', '', 'Each cell summarizes client elapsed_seconds / 60 for all 100 selected terminal requests in that setting, with sample SD using n−1. Timing includes server waiting, generation, streaming and network transfer within the request; local scheduling waits and retry backoff between requests are excluded.', '', '| Task/size | GPT-5.2 | Gemini 3.1 Pro Preview | Claude Opus 4.5 | DeepSeek V3.2 |', '|---|---:|---:|---:|---:|'])
    for task, scale, label in [('sudoku', '9x9', 'Sudoku 9×9'), ('maze', '30x30', 'Maze 30×30')] + [('sat', str(n), 'SAT n=' + str(n)) for n in [8, 16, 32, 64, 128, 256]]:
        values = []
        for model in models:
            s = groups[task, model, scale]
            values.append(f"{s['mean_minutes']:.3f} ± {s['sample_sd_minutes']:.3f}")
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines.extend(['', '## Terminal status, routing and provenance', '', 'Of 3,200 results, 2,647 are completed, 552 length_limited and one no_final_content; all have valid durations and 885 are correct. Scores and statuses of the first terminal model responses are retained separately. The additional calls used 2,545 attempts, including 1,285 transient execution failures; all 1,260 additional trials eventually obtained terminal model responses.', '', 'All four models use the 005 prompt and frozen reasoning configuration. Additional Gemini n8/n64/n128 calls use a fixed Vertex Flex route: each setting contains ten historical Standard and 90 additional Flex calls, with the same model and high/65,536 budget.', '', 'Reported cost across all additional attempts is $757.12035632: 1,578 attempts report cost and 967 lack usage. Costs are listed separately under the scope in [cost_accounting.json](cost_accounting.json).', '', 'Tables: [24 SAT settings](sat_accuracy.csv) · [32 selected settings](current_settings.csv) · [Full-precision duration statistics](timing_summary.csv) · [Per-call durations](timing.csv).', '', 'Frozen sources: [Merged report](sources/merged_report.json.gz) · [Trial manifest](sources/TOPUP_MANIFEST.json.gz) · [Completion record](sources/completion.json) · [Source hashes](source_manifest.json).', '', 'See the [README](README.md) for complete questions and final responses. Match task/model/scale in current_settings to selected_cases, then join source_artifact_index by job_id to locate the full thinking in the original local result.json and events.jsonl. This package stores final answers and source indices.', '', 'Run `python -B data/llm_sat_100trials_20261015/verify.py` in the historical workspace to recompute scores, tables, durations and hashes offline; see [verification.json](verification.json).', ''])
    (HERE / 'REPORT.md').write_text('\n'.join(lines))

def update_manifest():
    owned = {'build_package.py', 'verify.py', 'frozen_grading.py', 'README.md', 'REPORT.md', 'selected_cases.csv', 'timing.csv', 'current_settings.csv', 'sat_accuracy.csv', 'timing_summary.csv', 'final_answers.jsonl.gz', 'questions.jsonl.gz', 'source_artifact_index.jsonl.gz', 'new_attempts.csv', 'cost_accounting.json', 'excluded_historical_cases.csv', 'cohort_summary.json', 'source_manifest.json'}
    files = [p for p in HERE.rglob('*') if p.is_file() and (p.relative_to(HERE).parts[0] == 'sources' or (p.parent == HERE and p.name in owned))]
    write_json('package_manifest.json', dict(files=[dict(path=str(p.relative_to(HERE)), sha256=sha(p.read_bytes()), bytes=p.stat().st_size) for p in sorted(files)], excluded=['package_manifest.json', 'verification.json', 'deepseek_flops/', 'plot_figures.py', 'report_comparison.csv', 'figure verification records', '__pycache__/']))
if __name__ == '__main__':
    main()
