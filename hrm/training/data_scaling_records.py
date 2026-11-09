"""Data identity and native fixed-update selection checks for E10."""
import hashlib
import json
from pathlib import Path


def ordered_dataset_identity(dataset):
    digest=hashlib.sha256()
    for row in dataset:
        digest.update(json.dumps([row['question'],row['answer']],separators=(',',':')).encode())
        digest.update(b'\n')
    return dict(samples=len(dataset),ordered_sha256=digest.hexdigest())


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_e10_training(directory, plan, seed):
    """Validate the supplied native training evidence, not unobserved execution."""
    directory=Path(directory)
    def read(name):return json.loads((directory/name).read_text())
    config=read('model_config.json');summary=read('training_summary.json');inputs=read('input_provenance.json')
    records=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines() if line.strip()]
    required=dict(total_steps=83200,num_evals=20,cycles_per_data=16,eval_cycles=16,
                  eval_stable_cycles=16,local_batch_size=96,lr=.0001,lr_warmup_steps=2000,
                  lr_min_ratio=1.,weight_decay=1.,beta1=.9,beta2=.95,ema=.999)
    if any(config.get(k)!=v for k,v in required.items()):
        raise ValueError('Training configuration differs from the E10 schedule')
    if seed not in config['seeds'] or inputs['seed']!=seed or inputs['world_size']!=8:
        raise ValueError('Training seed or world size differs from E10')
    if config['data']['eval_split']!='test' or config['data']['train_size']!=1000 or config['data']['test_size']!=20000:
        raise ValueError('Training split sizes differ from E10')
    if not config['data'].get('augment'):
        raise ValueError('E10 requires training Sudoku augmentation')
    if inputs['train']['samples']!=1000 or inputs['test']['samples']!=20000:
        raise ValueError('Recorded dataset sizes differ from E10')
    if (plan['split'],plan['samples'],plan['dataset_order_sha256'])!=('test',20000,inputs['test']['ordered_sha256']):
        raise ValueError('Curve cohort differs from the checkpoint-selection cohort')
    # Both saved model configurations must describe the same numerical model.
    from hrm.evaluation.config import EvalConfig
    metadata=dict(vocab_size=10,seq_len=82,is_causal=False)
    if EvalConfig(**config).model_config_dict(metadata)!=EvalConfig(**plan['config']).model_config_dict(metadata):
        raise ValueError('Curve architecture differs from the trained configuration')
    from hrm.paths import repository_root
    family={'hrm.models.hrm@HRM':'hrm','hrm.models.rt@RecurrentTransformer':'rt'}.get(config['arch']['name'])
    if family is None:raise ValueError('Unsupported E10 training architecture')
    template=json.loads((repository_root()/f'exp/sudoku/configs/stable_stopping/{family}.json').read_text())
    if EvalConfig(**config).model_config_dict(metadata)!=EvalConfig(**template).model_config_dict(metadata):
        raise ValueError('Trained model dimensions or schedule differ from E10')
    if inputs['data']!=config['data']:
        raise ValueError('Recorded data policy differs from training configuration')
    if config['arch']['name']!=plan['config']['arch']['name']:
        raise ValueError('Curve model class differs from training')
    expected_steps=list(range(4160,83201,4160))
    if [r['optimizer_steps'] for r in records]!=expected_steps:
        raise ValueError('Missing, repeated or out-of-order training evaluations')
    for index,row in enumerate(records,1):
        step=index*4160
        expected=dict(evaluation=index,seed=seed,world_size=8,global_batch_size=768,
                      cycles_per_data=16,trained_examples=((step-1)//16+1)*768,
                      optimizer_example_exposures=step*768,interval_trained_examples=260*768)
        if any(row.get(k)!=v for k,v in expected.items()):
            raise ValueError('Training evaluation accounting differs from E10')
        score=row['eval']['test']
        if (score['total']!=20000 or type(score['correct']) is not int or not 0<=score['correct']<=20000
            or score['exact_match']!=score['correct']/20000 or score['max_cycles']!=16
            or score['stable_cycles']!=16 or score['mean_cycles']!=16):
            raise ValueError('Training evaluation has an incompatible metric or stopping budget')
    best=max(records,key=lambda r:(r['eval']['test']['exact_match'],r['optimizer_steps']))
    expected=dict(status='complete',seed=seed,total_steps=83200,num_evals=20,optimizer_steps=83200,
                  trained_examples=5200*768,optimizer_example_exposures=83200*768)
    if any(summary.get(k)!=v for k,v in expected.items()) or summary['best_eval']!=best or summary['final_eval']!=records[-1]:
        raise ValueError('Training completion/selection summary differs from the full log')
    # Resolve adjacent weights by basename so a copied run remains portable.
    checkpoint=directory/Path(best['checkpoint']).name
    digest=file_hash(checkpoint)
    if digest!=best.get('checkpoint_sha256') or digest!=plan['checkpoint_sha256']:
        raise ValueError('Evaluated weights are not the selected latest-tie checkpoint')
    return dict(optimizer_step=best['optimizer_steps'],checkpoint_sha256=digest,
                selection_correct=best['eval']['test']['correct'],selection_total=20000,
                selection_rule='Highest fixed-16 test accuracy; latest optimizer-step tie',
                train=inputs['train'],test=inputs['test'],
                evidence_hashes={name:file_hash(directory/name) for name in
                    ('model_config.json','training_summary.json','metrics.jsonl','input_provenance.json')})
