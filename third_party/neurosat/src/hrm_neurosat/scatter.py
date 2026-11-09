import torch

def scatter_sum(src, index, dim=0, dim_size=None):
    assert dim == 0
    size = int(index.max()) + 1 if dim_size is None else dim_size
    return src.new_zeros((size, *src.shape[1:])).index_add(0, index, src)

def scatter_mean(src, index, dim=0, dim_size=None):
    out = scatter_sum(src, index, dim, dim_size)
    count = scatter_sum(torch.ones_like(src), index, dim, dim_size)
    return out / count.clamp_min(1)
