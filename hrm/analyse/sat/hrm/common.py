"""Compatibility API backed by the shared SAT task implementation."""
import json
from hrm.paths import repo_path, work_path
from hrm.tasks.sat_flow import SATTask, sat_batch, sat_satisfies
from hrm.training.optimizer import AdamATan2
P = work_path('sat/hrm')
CFG = json.loads(repo_path('data/sat/hrm/training_config.json').read_text())
CFG['model_family'] = 'hrm'
_task = SATTask(CFG, repo_path('data/sat/hrm'))
model, bind, arrays, batch, ce_cases = (_task.model, _task.bind, _task.arrays, _task.batch, _task.ce_cases)
def write(path, obj):
    from pathlib import Path
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
