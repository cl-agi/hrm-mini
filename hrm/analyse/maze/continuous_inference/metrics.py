from collections import deque
import numpy as np
import torch
import torch.nn.functional as F

def geometry(x):
    boards=x[:,1:].reshape(-1,30,30);ds=[];dg=[]
    for b in boards:
        assert (b==3).sum()==1 and (b==4).sum()==1
        distances=[]
        for token in [3,4]:
            d=np.full((30,30),10000,np.int64);r,c=np.argwhere(b==token)[0];d[r,c]=0;q=deque([(int(r),int(c))])
            while q:
                r,c=q.popleft()
                for rr,cc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                    if 0<=rr<30 and 0<=cc<30 and b[rr,cc]!=1 and d[rr,cc]==10000:
                        d[rr,cc]=d[r,c]+1;q.append((rr,cc))
            distances.append(d)
        ds.append(distances[0]);dg.append(distances[1])
    return np.stack(ds),np.stack(dg)

def neighbors(z,pad_value=0):
    p=F.pad(z,(1,1,1,1),value=pad_value)
    return torch.stack([p[:,:-2,1:-1],p[:,2:,1:-1],p[:,1:-1,:-2],p[:,1:-1,2:]],-1)

def metrics(pred,b):
    x=b['x'][:,1:].view(-1,30,30);y=pred[:,1:].view(-1,30,30)
    selected=(y==5)|(x==3)|(x==4)
    ds,dg=b['ds'],b['dg'];distance=ds.masked_fill(x!=4,0).sum((1,2))
    on_dag=(ds+dg)==distance[:,None,None]
    # Distances strictly increase along the selected path; no disconnected cycles.
    nb=neighbors(selected);dn=neighbors(ds,10000)
    pred_count=(nb&(dn==ds[:,:,:,None]-1)).sum(-1)
    succ_count=(nb&(dn==ds[:,:,:,None]+1)).sum(-1)
    good_degree=(pred_count==(x!=3).long())&(succ_count==(x!=4).long())
    structure=((~selected)|(on_dag&good_degree)).all((1,2))&(selected.sum((1,2))==distance+1)
    fmt=torch.where(x==2,(y==2)|(y==5),y==x).all((1,2))&(pred[:,0]==0)
    return structure&fmt,(pred==b['y']).all(1)

def cpu_valid(pred,x):
    # Independent graph traversal, no use of the DAG-degree implementation.
    ds,dg=geometry(x[None]);board=x[1:].reshape(30,30);p=pred[1:].reshape(30,30)
    if pred[0]!=0 or not np.where(board==2,(p==2)|(p==5),p==board).all():return False
    nodes=set(map(tuple,np.argwhere((p==5)|(board==3)|(board==4))))
    start=tuple(np.argwhere(board==3)[0]);goal=tuple(np.argwhere(board==4)[0]);seen={start};q=deque([start])
    while q:
        r,c=q.popleft()
        for rc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if rc in nodes and rc not in seen:seen.add(rc);q.append(rc)
    return goal in seen and len(seen)==len(nodes)==int(ds[0][goal])+1
