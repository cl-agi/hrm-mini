from hrm.paths import analysis_directory
from hrm.paths import repo_path, work_path
from pathlib import Path
from pathlib import Path
import json, hashlib, csv, collections, shutil
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
p = analysis_directory(__file__)
assert shutil.disk_usage('/').free > 15 * 1024 ** 3
old = Path(str(work_path('sudoku_scaling')))
meta = json.load(open(old / 'dataset_manifest.json'))
files = [Path(x['filename']) for x in meta['splits']['train']['cache_files']]
tables = []
for f in files:
    with pa.memory_map(str(f), 'r') as stream:
        tables.append(pa.ipc.open_stream(stream).read_all())
t = pa.concat_tables(tables)
N = len(t)
assert N == 3831994
sources = t['source'].to_numpy()
names = sorted(set(sources))
pool = {s: np.flatnonzero(sources == s) for s in names}
rng = np.random.default_rng(42)
for ids in pool.values():
    rng.shuffle(ids)
sizes = [1000000]
previous = {s: 0 for s in names}
records = []
testsrc = Path(str(work_path('hrm_attention_missing_ablations/data/test_hard.csv')))
test = list(csv.DictReader(testsrc.open()))
assert len(test) == 20000
test_questions = {r['question'].replace('0', '.') for r in test}
overlaps = []
for batch in t.select(['question']).to_batches(max_chunksize=100000):
    overlaps.extend((q for q in batch.column(0).to_pylist() if q.replace('0', '.') in test_questions))
assert not overlaps, f'Full train overlaps official test: {len(overlaps)}'
for n in sizes:
    quotas = {s: n * len(pool[s]) // N for s in names}
    rem = n - sum(quotas.values())
    order = sorted(names, key=lambda s: (-(n * len(pool[s]) % N), s))
    for s in order[:rem]:
        quotas[s] += 1
    assert sum(quotas.values()) == n and all((quotas[s] >= previous[s] for s in names))
    previous = quotas
    indices = np.concatenate([pool[s][:quotas[s]] for s in names])
    np.random.default_rng(42 + n).shuffle(indices)
    d = p / 'data' / f'n{n}'
    d.mkdir(parents=True, exist_ok=True)
    assert not (d / 'manifest.json').exists(), 'Do not silently replace a prepared dataset'
    pacsv.write_csv(t.select(['question', 'answer']).take(pa.array(indices)), d / 'train.csv')
    shutil.copy2(testsrc, d / 'test_hard.csv')
    np.save(d / 'selected_full_indices.npy', indices)
    hashes = {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in ['train.csv', 'test_hard.csv', 'selected_full_indices.npy']}
    row = dict(train_size=n, source_counts=quotas, full_source_counts={s: len(pool[s]) for s in names}, allocation='Hamilton largest remainder of full source proportions', sampling_seed=42, nested_source_prefixes=True, source_files=[str(f) for f in files], test_source=str(testsrc), exact_fulltrain_test_overlap=0, hashes=hashes)
    (d / 'manifest.json').write_text(json.dumps(row, indent=2))
    records.append(row)
    print(n, quotas, flush=True)
(p / 'data_manifest.json').write_text(json.dumps(dict(full_size=N, source_revision=meta['source_hub_revision'], datasets=records), indent=2))
