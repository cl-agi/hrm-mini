"""Resolve the model identifiers used by experiment configurations."""
import importlib

MODULES = {"dataset.sudoku_blank": "hrm.tasks.sudoku_blank_difficulty_1k", 'arch.hrm_flow_bias': 'hrm.models.hrm_flow_bias', 'arch.rt_flow_bias':'hrm.models.rt_flow_bias', 'arch.hrm': 'hrm.models.hrm', 'arch.rt': 'hrm.models.rt', 'arch.layers': 'hrm.models.layers', 'arch.hrm_ablations': 'hrm.models.ablations', 'arch.hrm_truncate': 'hrm.models.truncated', 'arch.hrm_multiscale': 'hrm.models.multiscale', 'arch.ar_transformer': 'hrm.models.autoregressive', 'arch.mae': 'hrm.models.feedforward', 'arch.mae_deep_supervision': 'hrm.models.deep_supervision', 'arch.mae_ds_shared': 'hrm.models.shared_deep_supervision', 'arch.transformer': 'hrm.models.transformer', 'arch.trm': 'hrm.models.trm', 'arch.distributed_kl': 'hrm.models.distributed_kl', 'arch.stochastic_hrm': 'hrm.models.stochastic_hrm', 'arch.stochastic_rt': 'hrm.models.stochastic_rt', 'arch.hrm_act': 'hrm.models.hrm_act', 'arch.rt_act': 'hrm.models.rt_act', 'dataset.sudoku': 'hrm.tasks.sudoku', 'dataset.maze': 'hrm.tasks.maze'}

def resolve_module(name):
    if name.startswith(("arch.hrm.", "dataset.hrm.")):
        name = name.split(".", 1)[1]
    return MODULES.get(name, name)

def load_class(identifier):
    module, name = identifier.split("@")
    return getattr(importlib.import_module(resolve_module(module)), name)

def model_class(identifier, config):
    """Disambiguate historical `hrm@HRM` configs by their declared variant fields."""
    module, classname = identifier.split('@')
    module = resolve_module(module)
    if module == 'hrm.models.hrm':
        variants = []
        if 'bp_steps' in config: variants.append('truncated')
        if any(k in config for k in ['dual_module','hh_link','sym_io']): variants.append('ablations')
        if 'readout' in config: variants.append('readout')
        if len(variants)>1: raise ValueError('Combined historical HRM variants require an explicit model module')
        if variants: module = 'hrm.models.' + variants[0]
    return load_class(module+'@'+classname)
