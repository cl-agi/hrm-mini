"""Transport each Sudoku example and its recurrent states to a fresh coordinate frame."""
import torch


def sample_mapping(batch, device, generator):
    def axis():
        groups = torch.rand((batch, 3), device=device, generator=generator).argsort(-1)
        inside = torch.rand((batch, 3, 3), device=device, generator=generator).argsort(-1)
        return (3 * groups.unsqueeze(-1) + inside).flatten(1)
    rows, cols = axis(), axis()
    cells = (rows.unsqueeze(-1) * 9 + cols.unsqueeze(1)).flatten(1) + 1
    return torch.cat((torch.zeros((batch, 1), device=device, dtype=torch.long), cells), 1)


def transport(x, y, carry, mapping):
    """mapping[new token] = old token; BOS=0, digits unchanged."""
    return (x.gather(1, mapping), y.gather(1, mapping),
            {key: state.gather(1, mapping.unsqueeze(-1).expand(-1, -1, state.shape[-1]))
             for key, state in carry.items()})


def augment_outer(x, y, carry, generator):
    mapping = sample_mapping(x.shape[0], x.device, generator)
    return transport(x, y, carry, mapping)


def augment_active(x, y, carry, active, generator):
    """Transport surviving ACT slots; completed slots stay unchanged until refill."""
    mapping = sample_mapping(x.shape[0], x.device, generator)
    identity = torch.arange(x.shape[1], device=x.device).expand_as(mapping)
    mapping = torch.where(active[:, None], mapping, identity)
    return transport(x, y, carry, mapping)
