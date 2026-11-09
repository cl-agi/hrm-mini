"""Export only 2D fields from a cache. Never load its conformer-angle arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
manifest=json.loads(a.manifest.read_text());entries=manifest if isinstance(manifest,list) else manifest['entries']
requested=[row['id'] for row in entries]
assert len(set(requested)) == len(requested)
with np.load(a.cache,allow_pickle=False) as f:
    # NPZ loads members lazily. Do not enumerate/copy f.files or read angles.
    ids=f['ids'];graphs=f['graph_smiles'];lengths=f['lengths']
lookup={str(mid):i for i,mid in enumerate(ids)}
conditions=[dict(id=mid,graph_smiles=str(graphs[lookup[mid]]),length=int(lengths[lookup[mid]])) for mid in requested]
result=dict(conditions=conditions,cache_sha256=hashlib.sha256(a.cache.read_bytes()).hexdigest(),
            selection_manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
            cache_fields_read=['ids','graph_smiles','lengths'],reference_angles_read=False)
if a.out.exists():raise ValueError('Refusing to replace conditions')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
