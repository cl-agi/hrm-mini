from hrm.paths import analysis_directory
from pathlib import Path
import hashlib,json,shutil,sqlite3,time
P=analysis_directory(__file__)
HRM=P.parent/'sat_public_starts64_outer_20261025_v1'
source=P.parent/'sat1024_rt_ce_20261009'
assert shutil.disk_usage('/').free>15*1024**3
assert shutil.disk_usage(P).free>1024**4
d=P/'n1024';d.mkdir(exist_ok=True)
assert not (d/'config.json').exists()
cfg=json.loads((HRM/'n1024/config.json').read_text())
checkpoint=source/'weights/s24576.pt'
sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert sha=='69c6fe74467aa42f79a94fd6b84928f1ecedeae5a769a0b4e420f150a72f7bcd'
snapshot=d/'source';snapshot.mkdir()
for name in ('common.py','encoding.py','config.json'):
    shutil.copy2(source/name,snapshot/name)
shutil.copytree(source/'snapshot',snapshot/'snapshot',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
cfg.update(model='rt',source_run=str(snapshot),original_source_run=str(source),
    checkpoint=str(checkpoint),checkpoint_step=24576,checkpoint_sha256=sha,
    seed=int(time.time()),created=time.time(),gpus='0,1,2,3,4,5,6,7',workers=8,
    checkpoint_selection='Previously selected validation-best raw checkpoint; no public-test selection',
    seed_comparison='Independent timestamp seed; same formulas and start count, not paired trajectory seeds')
cfg['source_sha256']={str(f.relative_to(snapshot)):hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot.rglob('*') if f.is_file()}
for density,expected in cfg['data_hashes'].items():
    src=HRM/f'n1024/data_{density}.npz'
    assert hashlib.sha256(src.read_bytes()).hexdigest()==expected
    shutil.copy2(src,d/src.name)
(d/'config.json').write_text(json.dumps(cfg,indent=2))
db=sqlite3.connect(d/'queue.sqlite')
db.execute('CREATE TABLE tasks (id INTEGER PRIMARY KEY,density TEXT,case_id INTEGER,status TEXT,worker INTEGER,covered_outer INTEGER NOT NULL DEFAULT 0)')
for i in range(max(cfg['counts'].values())):
    for density,count in cfg['counts'].items():
        if i<count:db.execute('INSERT INTO tasks(density,case_id,status) VALUES (?,?,?)',(density,i,'pending'))
db.commit()
(P/'stage.json').write_text(json.dumps(dict(outer=1024,auto_advance=False,next_stage=2048,reason='RT first catches up with HRM at 1024 outer'),indent=2))
print(json.dumps(dict(seed=cfg['seed'],cases=sum(cfg['counts'].values()),checkpoint=str(checkpoint))))
