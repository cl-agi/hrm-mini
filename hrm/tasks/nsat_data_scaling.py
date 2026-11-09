"""Variable-width CNF adapter extending origin/SAT:dataset/sat.py.

Preserves the branch's literal-node polarity convention and all mask links,
including the dense clause-to-clause block. Zero is padding, never a literal.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


def pad_clause(clauses_in, max_clauses):
    if not clauses_in or any(not clause for clause in clauses_in):
        raise ValueError('Nonempty clauses required; an empty clause is UNSAT')
    if len(clauses_in) > max_clauses:
        raise ValueError('max_clauses is too small')
    return [list(clauses_in[i % len(clauses_in)]) for i in range(max_clauses)]


def pack_clauses(formulas, max_clauses=None):
    if not formulas:
        raise ValueError('Empty batch')
    maximum = max(len(f) for f in formulas) if max_clauses is None else max_clauses
    padded = [pad_clause(f, maximum) for f in formulas]
    width = max(len(c) for f in padded for c in f)
    result = torch.zeros((len(formulas), maximum, width), dtype=torch.long)
    for b, formula in enumerate(padded):
        for j, clause in enumerate(formula):
            if any(lit == 0 for lit in clause):
                raise ValueError('Zero is reserved for padding, not a DIMACS literal')
            result[b, j, :len(clause)] = torch.tensor(clause)
    return result


def expand_not(variable):
    return torch.stack([variable, ~variable], dim=-1).flatten(-2)


def get_clauses_state(clauses, answer):
    if clauses.abs().max() >= answer.shape[-1]:
        raise ValueError('Literal index outside assignment (assignment must include dummy variable 0)')
    values = answer.bool().gather(-1, clauses.abs().flatten(1)).view_as(clauses)
    return (torch.where(clauses > 0, values, ~values) & clauses.ne(0)).any(-1)


def is_valid_nsat(clauses, answer):
    return get_clauses_state(clauses, answer).all(-1)


@dataclass
class AttentionMaskSpec:
    """Exact branch mask as incidence links plus an implicit dense clause block.

    A symbolic spec permits measuring large circuits without allocating S**2.
    It does not make the branch's all-to-all clause attention computationally sparse.
    """
    clauses: torch.Tensor
    num_variables: int

    @property
    def max_clauses(self):
        return self.clauses.shape[1]

    @property
    def seq_len(self):
        return 2*self.num_variables+self.max_clauses+1

    @property
    def dense_bytes(self):
        return self.clauses.shape[0]*self.seq_len**2  # torch.bool = one byte

    def materialize(self, max_bytes=2**30):
        if self.dense_bytes > max_bytes:
            raise MemoryError(f'Exact SAT branch mask needs {self.dense_bytes/2**30:.2f} GiB '
                              f'for {self.seq_len:,} tokens, excluding attention/activations')
        c = self.clauses
        if c.abs().max() > self.num_variables:
            raise ValueError('Literal out of range')
        b, m, width = c.shape
        s = self.seq_len
        mask = torch.zeros((b, s, s), dtype=torch.bool, device=c.device)
        batch_ids = torch.arange(b, device=c.device)[:, None, None].expand_as(c)
        clause_nodes = (torch.arange(m, device=c.device)+2*self.num_variables+1)[None, :, None].expand_as(c)
        literal_nodes = torch.where(c > 0, 2*c, -2*c-1)
        active = c.ne(0)
        bi, li, ci = batch_ids[active], literal_nodes[active], clause_nodes[active]
        mask[bi, li, ci] = True
        mask[bi, ci, li] = True
        diagonal = torch.arange(2*self.num_variables+1, device=c.device)
        mask[:, diagonal, diagonal] = True
        neg = torch.arange(self.num_variables, device=c.device)*2+2
        mask[:, neg, neg-1] = True
        mask[:, neg-1, neg] = True
        mask[:, 2*self.num_variables+1:, 2*self.num_variables+1:] = True
        mask[:, :, 0] = True
        mask[:, 0, :] = True
        return mask[:, None]


def build_attn_mask(clauses, num_variables, max_clauses=None, max_bytes=2**30):
    if max_clauses is not None and clauses.shape[1] != max_clauses:
        raise ValueError('Pad clauses before constructing the mask')
    return AttentionMaskSpec(clauses, num_variables).materialize(max_bytes)


def collate_fn(batch, num_variables, max_clauses=None, augment=False, generator=None,
               materialize_mask=True, max_mask_bytes=2**30):
    if augment:
        raise NotImplementedError('SAT branch augmentation is not implemented')
    clauses = pack_clauses([item['clauses'] for item in batch], max_clauses)
    answers = torch.tensor([[0]+list(item['assignment']) for item in batch]).long()
    if answers.shape[1] != num_variables+1 or not torch.all((answers == 0) | (answers == 1)):
        raise ValueError('Invalid assignment')
    if not is_valid_nsat(clauses, answers).all():
        raise ValueError('Assignment does not satisfy CNF')
    b, m, _ = clauses.shape
    initial = F.pad(torch.randint(0, 2, (b, num_variables), generator=generator), (1, 0)).bool()
    x = torch.cat((torch.zeros(b, 1), expand_not(initial[:, 1:]).int()+1,
                   get_clauses_state(clauses, initial).int()+1), dim=1).long()
    y = torch.cat((torch.zeros(b, 1), expand_not(answers[:, 1:].bool()).int()+1,
                   torch.full((b, m), 2)), dim=1).long()
    y_mask = torch.zeros_like(x)
    y_mask[:, :2*num_variables+1] = 1
    spec = AttentionMaskSpec(clauses, num_variables)
    return dict(x=x, y=y, y_mask=y_mask, clauses=clauses, n=num_variables,
                attn_mask=spec.materialize(max_mask_bytes) if materialize_mask else spec)


def create_dataloader(split, batch_size, rank, world_size, dataset_name, num_variables,
                      max_clauses, augment=False, repeat=1, num_workers=1, seed=42):
    """Drop-in branch loader for HF datasets or local JSON/JSONL formula files.

    Each row has `clauses: list[list[int]]` and `assignment: list[int]`.
    No clause width is assumed. Large masks fail before allocation with a size estimate.
    """
    from functools import partial
    from pathlib import Path
    from torch.utils.data import DataLoader, DistributedSampler
    from datasets import load_dataset
    path = Path(dataset_name)
    if path.is_dir() and (path/f'{split}.jsonl').exists():
        dataset = load_dataset('json', data_files={split: str(path/f'{split}.jsonl')})[split]
    elif path.is_file() and path.suffix in ('.json', '.jsonl'):
        dataset = load_dataset('json', data_files={split: str(path)})[split]
    else:
        dataset = load_dataset(dataset_name)[split]
    is_train = 'train' in split
    dataset = dataset.repeat(repeat if is_train else 1)
    return DataLoader(dataset, batch_size=batch_size,
        collate_fn=partial(collate_fn,num_variables=num_variables,max_clauses=max_clauses,
                           augment=augment and is_train),
        sampler=DistributedSampler(dataset,rank=rank,num_replicas=world_size,
                                   shuffle=is_train,drop_last=True,seed=seed),
        drop_last=True,pin_memory=True,persistent_workers=num_workers>0,
        num_workers=num_workers,prefetch_factor=2 if num_workers else None), {
        'vocab_size': 3, 'seq_len': 2*num_variables+max_clauses+1, 'is_causal': False}
