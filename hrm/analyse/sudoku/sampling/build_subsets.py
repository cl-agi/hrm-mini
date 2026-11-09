from hrm.paths import analysis_directory
from pathlib import Path
import json,csv,hashlib,collections,statistics
import numpy as np
P=analysis_directory(__file__);x=json.loads((P/'provenance.json').read_text())
for key in ['new_1k','old_1k','old_100']:
 for r in x[key]:r['rating']=int(r['rating'])
def write(name,rows):
 out=P/name;out.mkdir(exist_ok=True)
 with (out/'train.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['source','question','answer','rating']);w.writeheader();w.writerows([{k:r[k] for k in w.fieldnames} for r in rows])
 return {'file':str((out/'train.csv').relative_to(P)),'sha256':hashlib.sha256((out/'train.csv').read_bytes()).hexdigest()}
def stats(rows):
 ratings=[r['rating'] for r in rows]
 return {'n':len(rows),'source_counts':dict(collections.Counter(r['source'] for r in rows)),'rating_mean':statistics.mean(ratings),'rating_quantiles':dict(zip(['p0','p25','p50','p75','p90','p95','p100'],[float(v) for v in np.percentile(ratings,[0,25,50,75,90,95,100])])),'givens_mean':statistics.mean(sum(c not in '.0' for c in r['question']) for r in rows)}
# Same method as full -> current 1k: shuffle without replacement, then select prefix.
rng=np.random.default_rng(42);indices=rng.permutation(1000)[:800];uniform=[x['new_1k'][int(i)] for i in indices]
# Legacy source-stratified extension with the existing 100 fixed as an anchor.
old=x['old_1k'];anchor=x['old_100'];counts=collections.Counter(r['source'] for r in old);ac=collections.Counter(r['source'] for r in anchor);big=counts.most_common(1)[0][0]
quota={s:round(800*n/1000) for s,n in counts.items()};quota[big]+=800-sum(quota.values());rng=np.random.default_rng(0);aq={r['question'] for r in anchor};strat=list(anchor)
for source in sorted(counts):
 pool=[r for r in old if r['source']==source and r['question'] not in aq];need=quota[source]-ac[source];assert 0<=need<=len(pool)
 strat.extend(pool[int(i)] for i in rng.permutation(len(pool))[:need])
strat=[strat[int(i)] for i in rng.permutation(len(strat))]
assert len(strat)==len({r['question'] for r in strat})==800
assert aq<={r['question'] for r in strat}<={r['question'] for r in old}
assert len(uniform)==len({r['question'] for r in uniform})==800
meta={'uniform_new1k_seed42':write('uniform_new1k_seed42',uniform),'source_stratified_legacy1k_seed0_keep100':write('source_stratified_legacy1k_seed0_keep100',strat),'new_uniform_parent_indices':[int(i) for i in indices],'legacy_source_quotas':quota,'new_uniform_overlap_old_prefix800':len({r['question'] for r in uniform}&{r['question'] for r in x['new_1k'][:800]}),'stats':{name:stats(rows) for name,rows in [('legacy1k',old),('legacy100',anchor),('current1k',x['new_1k']),('current_prefix800',x['new_1k'][:800]),('current_added200',x['new_1k'][800:]),('current_uniform800',uniform),('legacy_stratified800',strat)]}}
(P/'subset_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
(P / 'prepare_data_original.py').write_text(x['prepare_script']);(P / 'build_tiny_original.py').write_text(x['legacy_subset_script'])
for k,v in meta['stats'].items():print(k,json.dumps(v))
print('uniform overlap with prefix',meta['new_uniform_overlap_old_prefix800'])
