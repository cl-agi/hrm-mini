"""Read evaluation settings independently of training-controller fields."""
import json
from pathlib import Path
from typing import Any
import pydantic
from hrm.config import read_config
from hrm.training.trainer import ArchConfig, DataConfig
from hrm.paths import repo_path, checkpoint_config, dataset_path
from hrm.registry import resolve_module

class EvalDataConfig(DataConfig):
    eval_split: str = 'test_hard'

class EvalConfig(pydantic.BaseModel):
    arch: ArchConfig
    data: EvalDataConfig
    transformer: dict[str, Any] = {}
    cycles_per_data: int
    eval_cycles: int | None = None
    eval_stable_cycles: int = 0
    local_batch_size: int
    seeds: list[int] = []
    def model_config_dict(self, metadata):
        config = self.transformer | (self.arch.__pydantic_extra__ or {}) | metadata
        for field in ['num_layers','hidden_size','intermediate_size','head_dim']:
            if field in config:config[field]=int(config[field])
        return config

def load_config(checkpoint, config=None, *, streaming=False, dataset=None):
    path=Path(config) if config else checkpoint_config(checkpoint)
    value=read_config(path)
    data=value['data']
    data.setdefault('eval_split', value.get('best_metric_split', 'test_hard'))
    if streaming:
        data.pop('repeat',None)
        data['name']='hrm.tasks.sudoku_stable16'
    elif 'transformer' in value:
        data['name']='hrm.tasks.sudoku_mixer'
    else:
        name=data['name'];data['name']=resolve_module(name if name.startswith(('dataset.','hrm.')) else 'dataset.'+name)
    if dataset is not None:
        # Override before consulting any saved or archived dataset location.
        if str(dataset) in ('sapientinc/sudoku-extreme','sapientinc/sudoku-extreme-1k'):
            data.pop('prepared_data_path',None)
            data['dataset_name']=str(dataset)
        else:
            location=str(dataset_path(dataset))
            if data.get('prepared_data_path'):data['prepared_data_path']=location
            else:data['dataset_name']=location
    elif data.get('prepared_data_path'):
        data['prepared_data_path']=str(dataset_path(data['prepared_data_path']))
    elif data.get('dataset_name'):
        data['dataset_name']=str(dataset_path(data['dataset_name']))
    else:
        from hrm.training.data import resolve_config
        data=resolve_config(value)['data']
        value['data']=data
    return EvalConfig(**value)
