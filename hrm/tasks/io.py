"""Preserve named splits when loading the supplied CSV datasets locally."""
from pathlib import Path
from datasets import Features, Value, load_dataset as hf_load_dataset

def load_dataset(dataset_name, *args, split=None, features=None, **kwargs):
    if str(dataset_name).startswith('hf://'):
        dataset_name=str(dataset_name)[5:].removeprefix('datasets/')
        return hf_load_dataset(dataset_name,*args,split=split,features=features,**kwargs)
    if dataset_name in ('sapientinc/sudoku-extreme','sapientinc/sudoku-extreme-1k'):
        kwargs.setdefault('revision', {'sapientinc/sudoku-extreme':'58942f96baeb572ca3127e2a9e9c70f330783d6b','sapientinc/sudoku-extreme-1k':'48941acb15ff8cc57f6fb355dded588ef6c3c2b8'}[dataset_name])
        if features is not None:
            # Upstream CSVs also contain source/rating columns. Project after loading
            # instead of asking datasets to cast a four-column table to two columns.
            schema=Features({'source': Value('string'), 'question': Value('string'),
                             'answer': Value('string'), 'rating': Value('int64')})
            result=hf_load_dataset(dataset_name,*args,split=split,features=schema,**kwargs)
            result=result.select_columns(list(features))
            return result if result.features == features else result.cast(features)
    path=Path(dataset_name)
    file=path/(split+'.csv') if split is not None else None
    if path.is_file(): file=path
    if file is not None and file.is_file():
        if features is not None:kwargs.setdefault('usecols',list(features))
        result=hf_load_dataset('csv',data_files={split:str(file)},split=split,features=features,*args,**kwargs)
        if features is not None:
            result=result.select_columns(list(features)).cast(features)
        return result
    return hf_load_dataset(dataset_name,*args,split=split,features=features,**kwargs)
