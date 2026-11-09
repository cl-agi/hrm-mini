"""Small scalar evaluation records independent of optional experiment trackers."""
import json
from pathlib import Path


def record_evaluation(path, *, seed, step, split, correct, total, cycles, weights, stable_cycles=0, mean_cycles=None):
    if total<=0 or not 0<=correct<=total:raise ValueError('Invalid evaluation numerator/denominator')
    row=dict(seed=int(seed),optimizer_step=int(step),split=str(split),correct=int(correct),
             total=int(total),accuracy=correct/total,inference_cycles=int(cycles),weights=weights,stable_cycles=int(stable_cycles or 0))
    if mean_cycles is not None:row['mean_inference_cycles']=float(mean_cycles)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(row)+'\n')
    return row


def write_best_selection(checkpoint, evaluation_path, *, seed, step, split, cycles, weights):
    """Bind the actual saved best weights to their scalar selection evidence."""
    import hashlib
    checkpoint, evaluation_path = Path(checkpoint), Path(evaluation_path)
    content = evaluation_path.read_bytes()
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    rows = [r for r in rows if r['seed'] == seed and r['split'] == split
            and r['inference_cycles'] == cycles and r['weights'] == weights and r.get('stable_cycles',0) == 0]
    if not rows or len({r['optimizer_step'] for r in rows}) != len(rows):
        raise ValueError('Missing or repeated checkpoint-selection evaluation steps')
    for row in rows:
        if row['total'] <= 0 or not 0 <= row['correct'] <= row['total'] or row['accuracy'] != row['correct']/row['total']:
            raise ValueError('Invalid checkpoint-selection metric')
    if len({r['total'] for r in rows}) != 1:
        raise ValueError('Checkpoint selection mixes denominators')
    best=max(rows,key=lambda r:(r['accuracy'],-r['optimizer_step']))
    if best['optimizer_step'] != step:
        raise ValueError('Saved checkpoint step is not the earliest maximum evaluation')
    with checkpoint.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    receipt=dict(schema_version=1,checkpoint_file=checkpoint.name,checkpoint_sha256=digest,
        evaluation_file=evaluation_path.name,evaluation_prefix_bytes=len(content),
        evaluation_prefix_sha256=hashlib.sha256(content).hexdigest(),selected=best,
        rule='Highest fixed-cycle exact accuracy; earliest optimizer-step tie')
    path=checkpoint.with_suffix('.selection.json');temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');temporary.replace(path)
    return receipt


def verify_best_selection(receipt_path, *, checkpoint_sha256, seed, cycles=16, weights='EMA'):
    """Check a best-checkpoint receipt against the complete current scalar log."""
    import hashlib
    receipt_path=Path(receipt_path)
    receipt=json.loads(receipt_path.read_text())
    if receipt['checkpoint_sha256'] != checkpoint_sha256:
        raise ValueError('Selected checkpoint hash differs from evaluated weights')
    if Path(receipt['evaluation_file']).name != receipt['evaluation_file']:
        raise ValueError('Selection log must be adjacent to its receipt')
    content=(receipt_path.parent/receipt['evaluation_file']).read_bytes()
    prefix=content[:receipt['evaluation_prefix_bytes']]
    if len(prefix)!=receipt['evaluation_prefix_bytes'] or hashlib.sha256(prefix).hexdigest()!=receipt['evaluation_prefix_sha256']:
        raise ValueError('Original checkpoint-selection records changed')
    selected=receipt['selected']
    if selected not in [json.loads(line) for line in prefix.splitlines() if line.strip()]:
        raise ValueError('Selected evaluation is absent from the frozen log prefix')
    if (selected['seed'],selected['inference_cycles'],selected['weights'],selected.get('stable_cycles',0))!=(seed,cycles,weights,0):
        raise ValueError('Selection seed/budget/weight-state/stopping rule differs from the protocol')
    rows=[json.loads(line) for line in content.splitlines() if line.strip()]
    rows=[r for r in rows if (r['seed'],r['split'],r['inference_cycles'],r['weights'],r.get('stable_cycles',0))==
          (seed,selected['split'],cycles,weights,0)]
    if not rows or len({r['optimizer_step'] for r in rows})!=len(rows):
        raise ValueError('Missing or duplicate selection evaluations')
    if len({r['total'] for r in rows})!=1:
        raise ValueError('Selection denominators changed')
    for row in rows:
        if row['total']<=0 or not 0<=row['correct']<=row['total'] or row['accuracy']!=row['correct']/row['total']:
            raise ValueError('Invalid selection metric')
    if max(rows,key=lambda r:(r['accuracy'],-r['optimizer_step'])) != selected:
        raise ValueError('Later/better checkpoint exists or selected record is inconsistent')
    return selected
