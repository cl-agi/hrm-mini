"""Summarize model accuracy and uncertainty from correctness records.

Reads source files without changing them. Outputs correctness/accuracy only.
HRM: sample SD over training seeds. Qwen: descriptive sample SD over the
four within-puzzle generation positions. GPT and HRM+TTS: Wilson 95% CI
over evaluated puzzles. These uncertainty units remain explicitly distinct.
"""
from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist, mean, stdev, variance

def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def wilson(successes, n):
    z = NormalDist().inv_cdf(0.975)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (100 * (centre - radius), 100 * (centre + radius))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--qwen-root', type=Path, default=Path(str(work_path('hrm'))))
    parser.add_argument('--output-dir', type=Path, default=analysis_directory(__file__))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = []

    def source(path, role):
        path = path.resolve()
        sources.append({'path': str(path), 'role': role, 'sha256': digest(path), 'bytes': path.stat().st_size})
        return path
    summary = []

    def summarize(label, rates, kind, unit, n_observations, n_puzzles, n_correct=None):
        centre = mean(rates)
        sd = stdev(rates) if kind == 'sample SD' else None
        var = variance(rates) if kind == 'sample SD' else None
        ci = wilson(n_correct, n_puzzles) if kind == 'Wilson 95% CI' else (None, None)
        low, high = (centre - sd, centre + sd) if sd is not None else ci
        summary.append({'condition': label, 'mean_percent': centre, 'sd_percentage_points': sd, 'variance_percentage_points_squared': var, 'ci95_lower_percent': ci[0], 'ci95_upper_percent': ci[1], 'errorbar_lower_percent': low, 'errorbar_upper_percent': high, 'error_type': kind, 'effective_unit': unit, 'n_units': len(rates) if sd is not None else n_puzzles, 'n_puzzles': n_puzzles, 'n_outputs': n_observations, 'n_correct': n_correct})
    hrm_source = source(args.repo / 'data/original_ablation_20261020/redraw_v001/plotted_values.csv', 'HRM three training seeds at common training step 70720')
    rows = [r for r in read_csv(hrm_source) if r['model'] == 'tuned_HRM' and r['step'] == '70720']
    assert len(rows) == 1
    hrm = [{'training_seed': seed, 'training_step': 70720, 'accuracy_percent': float(rows[0][f'seed_{seed}_percent'])} for seed in (1, 2, 3)]
    write_csv(out / 'hrm_training_seed_accuracy.csv', hrm)
    summarize('HRM', [r['accuracy_percent'] for r in hrm], 'sample SD', 'training seed', None, None)
    gpt_source = source(args.repo / 'data/llm_sat_100trials_20261015/selected_cases.csv', 'Frozen selected GPT-5.2 Sudoku outcomes')
    rows = [r for r in read_csv(gpt_source) if r['task'] == 'sudoku' and r['model'] == 'openai/gpt-5.2']
    assert len(rows) == len({r['problem_sha256'] for r in rows}) == 100
    assert all((r['correct'] in {'True', 'False'} for r in rows))
    gpt = [{'case_id': r['case_id'], 'problem_sha256': r['problem_sha256'], 'correct': int(r['correct'] == 'True')} for r in rows]
    assert sum((r['correct'] for r in gpt)) == 24
    write_csv(out / 'gpt_sudoku_correctness.csv', gpt)
    summarize('GPT-5.2', [24.0], 'Wilson 95% CI', 'test puzzle', 100, 100, 24)
    qwen_correctness, qwen_repeats, qwen_puzzle_stats = ([], [], [])
    for label, relative, budget, expected_correct, ranks in [('Qwen base', 'diag/llm_rlvr_v2_4b_16k/step0', 16384, 1, 4), ('Qwen teacher-CoT SFT', 'diag/llm_cot_qwen3.5_4b_all1080/sft_budget65536', 65536, 182, 8)]:
        records = []
        for rank in range(ranks):
            path = source(args.qwen_root / relative / f'eval.rank{rank}.jsonl', label)
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                record = json.loads(line)
                if record['config'] == ['test_hard', 0, budget]:
                    assert len(record['samples']) == 4
                    records.append(record)
                    for slot, sample in enumerate(record['samples'], 1):
                        assert isinstance(sample['exact'], bool)
                        qwen_correctness.append({'condition': label, 'budget_tokens': budget, 'puzzle_index': record['index'], 'generation_position': slot, 'correct': int(sample['exact']), 'source_rank': rank, 'source_line': line_number})
        assert len(records) == 256 and {r['index'] for r in records} == set(range(256))
        successes = [sum((r['samples'][slot]['exact'] for r in records)) for slot in range(4)]
        assert sum(successes) == expected_correct
        rates = [100 * count / 256 for count in successes]
        for slot, (count, rate) in enumerate(zip(successes, rates), 1):
            qwen_repeats.append({'condition': label, 'generation_position': slot, 'n_puzzles': 256, 'n_correct': count, 'accuracy_percent': rate})
        puzzle_rates = [sum((s['exact'] for s in r['samples'])) / 4 for r in records]
        qwen_puzzle_stats.append({'condition': label, 'puzzle_rate_sample_variance_probability_squared': variance(puzzle_rates), 'puzzle_rate_sample_sd_percentage_points': 100 * stdev(puzzle_rates), 'mean_standard_error_percentage_points': 100 * stdev(puzzle_rates) / math.sqrt(256), 'n_puzzles': 256, 'outputs_per_puzzle': 4})
        summarize(label, rates, 'sample SD', 'within-puzzle generation position (descriptive)', 1024, 256, expected_correct)
    write_csv(out / 'qwen_correctness.csv', sorted(qwen_correctness, key=lambda r: (r['condition'], r['puzzle_index'], r['generation_position'])))
    write_csv(out / 'qwen_generation_position_accuracy.csv', qwen_repeats)
    write_csv(out / 'qwen_puzzle_level_statistics.csv', qwen_puzzle_stats)
    tts_source = source(args.repo / 'data/release_1k_tts_20260924/run_131072/per_puzzle.csv', 'Released HRM final board accuracy after 131072 outer cycles')
    rows = read_csv(tts_source)
    assert len(rows) == len({r['puzzle_index'] for r in rows}) == 2000
    assert all((r['exact_board81_final'] in {'0', '1'} for r in rows))
    tts = [{'puzzle_index': int(r['puzzle_index']), 'outer_cycles': 131072, 'correct': int(r['exact_board81_final'])} for r in rows]
    assert sum((r['correct'] for r in tts)) == 1963
    write_csv(out / 'hrm_tts_final_correctness.csv', tts)
    summarize('HRM+TTS', [98.15], 'Wilson 95% CI', 'test puzzle', 2000, 2000, 1963)
    order = {label: i for i, label in enumerate(['HRM', 'HRM+TTS', 'GPT-5.2', 'Qwen base', 'Qwen teacher-CoT SFT'])}
    summary.sort(key=lambda r: order[r['condition']])
    write_csv(out / 'summary.csv', summary)
    manifest = {'description': __doc__, 'repository_root': str(args.repo.resolve()), 'qwen_source_root': str(args.qwen_root.resolve()), 'sources': sources, 'methods': {'sample_sd_ddof': 1, 'wilson_confidence': 0.95, 'wilson_z': NormalDist().inv_cdf(0.975), 'Qwen': 'Four ordered samples per puzzle; these are not independent training seeds or separately launched runs.', 'HRM': 'Common training step 70720, three training seeds.', 'GPT_and_TTS': 'Each confidence interval quantifies puzzle-sample uncertainty for the single archived evaluation.'}, 'output_files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(out.glob('*.csv'))], 'script_sha256': digest(Path(__file__))}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
