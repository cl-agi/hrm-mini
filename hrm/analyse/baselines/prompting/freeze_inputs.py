"""Freeze one train sample using the established shuffle(42) protocol, offline."""
from hrm.paths import analysis_directory
from hrm.paths import repo_path
import csv
import json
import os
from pathlib import Path
import sys
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
import datasets
import numpy
from datasets import Dataset
ROOT = analysis_directory(__file__).parents[0]
pass
import hrm.analyse.baselines.frontier_llm.transport as transport

def write_frozen(path, text):
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Frozen file differs: {path}')
    else:
        path.write_text(text)

def jsonl(rows):
    return ''.join((json.dumps(row, ensure_ascii=False) + '\n' for row in rows))

def main():
    source = ROOT / 'sources/train.csv'
    with source.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    indices = list(Dataset.from_dict({'source_index': list(range(len(rows)))}).shuffle(seed=42, keep_in_memory=True).select(range(100))['source_index'])
    selected, references = ([], [])
    validator = transport.load_task_functions(ROOT / 'graders/sudoku.py')['validate']
    for sample_index, source_index in enumerate(indices):
        row = rows[source_index]
        problem = {'question': row['question']}
        if len(row['question']) != 81 or set(row['question']) - set('.123456789') or (not validator(row['question'], row['answer'])):
            raise ValueError(f'Invalid source puzzle or reference at row {source_index}')
        common = {'case_id': f'sudoku_train_sample{sample_index:03d}', 'problem_sha256': transport.problem_hash(problem), 'sample_index': sample_index, 'source_index': source_index}
        selected.append({**common, 'problem': problem})
        references.append({**common, 'task': 'sudoku', 'reference_answer': row['answer'], 'source_label': row['source'], 'source_rating': row['rating']})
    hashes = {row['problem_sha256'] for row in selected}
    if len(hashes) != 100:
        raise ValueError('Selected train sample contains duplicate puzzles')
    comparisons = {'prior_development_10': ROOT.parent / 'pilots/pilot_003_10_each_20260908/inputs/sudoku.jsonl', 'prior_test_hard_100': ROOT.parent / '013_deepseek_v4_pro_official/full_inputs/sudoku.jsonl'}
    overlaps = {}
    for name, path in comparisons.items():
        other = [json.loads(line) for line in path.read_text().splitlines() if line]
        matches = sorted(hashes & {row['problem_sha256'] for row in other})
        overlaps[name] = {'path': str(path), 'sha256': transport.file_hash(path), 'overlap_count': len(matches), 'problem_hashes': matches}
    output = ROOT / 'full_inputs/sudoku.jsonl'
    write_frozen(output, jsonl(selected))
    write_frozen(ROOT / 'full_inputs/reference_answers.jsonl', jsonl(references))
    metadata = (ROOT / 'sources/train.csv.metadata').read_text().splitlines()
    manifest = {'status': 'frozen_sudoku_train_100', 'dataset_id': 'sapientinc/sudoku-extreme-1k', 'dataset_url': 'https://huggingface.co/datasets/sapientinc/sudoku-extreme-1k', 'split': 'train', 'source_path': str(repo_path('data/sudoku/sudoku-extreme-1k/train.csv')), 'source_snapshot': str(source), 'source_sha256': transport.file_hash(source), 'local_huggingface_metadata_revision': metadata[0], 'source_rows': len(rows), 'selected_rows': 100, 'sampling_method': 'datasets.Dataset index table -> shuffle(seed=42, keep_in_memory=True) -> select(range(100))', 'sampling_seed': 42, 'source_indices_0based': indices, 'datasets_version': datasets.__version__, 'numpy_version': numpy.__version__, 'input_files': [{'file_key': 'sudoku', 'path': str(output), 'sha256': transport.file_hash(output), 'rows': 100}], 'reference_answers_sent_to_model': False, 'duplicate_problem_hashes_in_selected': 0, 'all_reference_answers_valid': True, 'overlap_checks': overlaps, 'selection_policy': 'One fixed random sample from all train rows; no difficulty, answer or model-performance filtering; overlapping development questions retained and identified.', 'api_called_during_selection': False}
    write_frozen(ROOT / 'full_inputs/manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    functions = transport.load_task_functions(ROOT / 'dataset/sudoku.py')
    write_frozen(ROOT / 'prompts/sudoku.txt', functions['get_prompt']())
    print(json.dumps({'selected_rows': len(selected), 'unique_puzzles': len(hashes), 'source_rows': len(rows), 'overlaps': {k: v['overlap_count'] for k, v in overlaps.items()}}))
if __name__ == '__main__':
    main()
