"""Prepare a new evaluation without changing the archived experiment."""
from hrm.paths import analysis_directory
from pathlib import Path
import hashlib
import json
import shutil
import sqlite3
import time

P = analysis_directory(__file__)
OLD = P.parent / 'public_sat_tts16k64_20261005'
assert shutil.disk_usage('/').free > 15 * 1024**3
assert shutil.disk_usage(P).free > 30 * 1024**3
for n in (512, 1024):
    dest = P / f'n{n}'
    dest.mkdir(exist_ok=True)
    assert not (dest / 'config.json').exists(), 'Do not overwrite a prepared run'
    old = json.loads((OLD / f'n{n}/config.json').read_text())
    seed = int(time.time())
    cfg = dict(old, seed=seed, starts=64, max_starts=64,
               outer=16384, budgets=[2**i for i in range(4, 15)],
               metric='any valid among fixed 64 starts through outer budget',
               seed_formula='seed + density_index*100000000 + official_id*64 + start_index',
               stopping='Once minimum first-valid outer is known, formula is solved at all later budgets',
               created=time.time(), host='unspecified')
    assert hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest() == cfg['checkpoint_sha256']
    for density, sha in cfg['data_hashes'].items():
        src = OLD / f'n{n}/data_{density}.npz'
        assert hashlib.sha256(src.read_bytes()).hexdigest() == sha
        shutil.copy2(src, dest / src.name)
    # Freeze the exact source used for inference, including model and encoding.
    source = Path(cfg['source_run'])
    snapshot = dest / 'source'
    snapshot.mkdir()
    for name in ('common.py', 'encoding.py', 'config.json'):
        shutil.copy2(source / name, snapshot / name)
    shutil.copytree(source / 'snapshot', snapshot / 'snapshot',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    cfg['original_source_run'] = cfg['source_run']
    cfg['source_run'] = str(snapshot)
    cfg['source_sha256'] = {str(f.relative_to(snapshot)): hashlib.sha256(f.read_bytes()).hexdigest()
                            for f in snapshot.rglob('*') if f.is_file()}
    (dest / 'config.json').write_text(json.dumps(cfg, indent=2))
    db = sqlite3.connect(dest / 'queue.sqlite')
    db.execute('CREATE TABLE tasks (id INTEGER PRIMARY KEY, density TEXT, case_id INTEGER, status TEXT, worker INTEGER)')
    # Interleave densities so neither group is delayed until the other finishes.
    for i in range(max(cfg['counts'].values())):
        for d, count in cfg['counts'].items():
            if i < count:
                db.execute('INSERT INTO tasks(density,case_id,status) VALUES (?, ?, ?)', (d, i, 'pending'))
    db.commit()
    print(json.dumps(dict(n_vars=n, seed=seed, cases=sum(cfg['counts'].values()))), flush=True)
    while int(time.time()) == seed:
        time.sleep(0.1)
