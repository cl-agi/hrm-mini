"""Global-batch mixture KL with differentiable cross-rank components."""
import math
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather

@torch.compiler.disable
def distributed_gaussian_kl(mu, log_std, sample, mode, valid, count):
    dtype=torch.float64 if mu.dtype==torch.float64 else torch.float32
    mu,ls,z=(v.to(dtype).flatten(1) for v in (mu,log_std,sample))
    world=dist.get_world_size();rank=dist.get_rank();b,d=mu.shape
    weight=valid.to(dtype)*world/count.to(dtype)
    individual=(.5*(mu.square()+(2*ls).exp()-1-2*ls).mean(1)*weight).sum()
    if mode=='single':return individual
    assert mode=='batch'
    means=torch.cat(all_gather(mu),0);logs=torch.cat(all_gather(ls),0)
    masks=[torch.empty_like(valid) for _ in range(world)];dist.all_gather(masks,valid)
    mask=torch.cat(masks)
    center=(means*mask[:,None]).sum(0,keepdim=True)/count
    m,v=means-center,z-center
    precision=(-2*logs).exp()
    quad=v.square()@precision.T-2*v@(m*precision).T+(m.square()*precision).sum(1)[None,:]
    cross=-.5*(quad+2*logs.sum(1)[None,:]+d*math.log(2*math.pi))
    own=-.5*(((z-mu).square()*(-2*ls).exp())+2*ls+math.log(2*math.pi)).sum(1)
    index=(torch.arange(b,device=mu.device)+rank*b)[:,None]
    cross=cross.scatter(1,index,own[:,None]).masked_fill(~mask[None,:],-torch.inf)
    correction=(torch.logsumexp(cross-own[:,None],1)-count.to(dtype).log())/d
    return individual+(correction*weight).sum()
