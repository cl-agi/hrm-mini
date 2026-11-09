"""Reconstruct the published graph-coloring CNF without archived NPZ files."""
import hashlib
from pathlib import Path
import numpy as np
from hrm import hub


def graph_to_cnf(row):
    k,n=row['num_colors'],row['num_vertices']
    flat=n==200 and k==3
    if not flat and (k not in (3,4) or n!=256):raise ValueError('Expected E43 generated or Flat200 graphs')
    edge_count=479 if flat else 512
    edges=np.asarray(row['edges'],dtype=np.int64)
    if edges.shape!=(edge_count,2) or np.any(edges<0) or np.any(edges>=n) or np.any(edges[:,0]==edges[:,1]):
        raise ValueError('Invalid E43 edges')
    if len(np.unique(np.sort(edges,axis=1),axis=0))!=edge_count:raise ValueError('Duplicate graph edges')
    sources=np.asarray(row['encoding_sources'] or [],dtype=np.int64)
    if sources.shape!=((256,) if k==3 and not flat else (0,)) or np.any(sources<0) or np.any(sources>=k*n):
        raise ValueError('Invalid copy-variable encoding map')
    clauses=[]
    if flat:
        for v in range(n):
            a,b,c=3*v+1,3*v+2,3*v+3
            clauses.extend([[-a,-b,-b],[-a,-c,-c],[-b,-c,-c],[a,b,c]])
    else:clauses=[[k*v+c+1 for c in range(k)] for v in range(n)]
    for u,v in edges:
        for c in range(k):clauses.append([-(k*int(u)+c+1)]+[-(k*int(v)+c+1)]*(k-1))
    for j,src in enumerate(sources):
        source=int(src)+1;destination=k*n+j+1
        clauses.extend([[-source,destination,destination],[source,-destination,-destination]])
    result=np.asarray(clauses,dtype=np.int32)
    if result.shape!=((2237 if flat else 2304),k):raise ValueError('Unexpected E43 clause count')
    return result


def load_data(colors,path=None,benchmark='generated'):
    if colors not in (3,4):raise ValueError('colors must be3 or4')
    if benchmark not in ('generated','flat200') or (benchmark=='flat200' and colors!=3):raise ValueError('Invalid graph benchmark/color combination')
    if path is None:rows=hub.dataset('sat','test_flat200' if benchmark=='flat200' else 'test_graph_coloring','graph_coloring')
    else:
        import pyarrow.parquet as pq
        rows=pq.read_table(hub.local_file(path)).to_pylist()
    selected=[r for r in rows if r['num_colors']==colors]
    if not selected:raise ValueError(f'No {colors}-color graphs')
    expected=100 if benchmark=='flat200' else 1000
    if any(r['num_vertices']!=(200 if benchmark=='flat200' else 256) for r in selected):raise ValueError('Wrong graph population for benchmark')
    if path is None and len(selected)!=expected:raise ValueError('Unexpected published graph count')
    ids=[r['id'] for r in selected]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate graph IDs')
    clauses=np.stack([graph_to_cnf(r) for r in selected])
    edges=np.asarray([r['edges'] for r in selected],dtype=np.int32)
    digest=hashlib.sha256()
    for rid,c,e in zip(ids,clauses,edges):
        digest.update(rid.encode()+b'\0');digest.update(c.astype('<i4').tobytes());digest.update(e.astype('<i4').tobytes())
    return dict(ids=ids,clauses=clauses,edges=edges,ordered_sha256=digest.hexdigest(),variables=600 if benchmark=='flat200' else 1024)
