"""Mixed OR/XOR constraints for the origin/SAT literal-factor attention graph.

Clause tokens encode operator, XOR target parity, and current satisfaction.
XOR repetitions cancel; negative literals toggle the target parity. The graph
therefore contains each remaining XOR variable exactly once. Zero is padding.
"""
from functools import partial
from pathlib import Path

import torch

from hrm.tasks.nsat_data_scaling import AttentionMaskSpec, expand_not

OR, XOR = 0, 1
VOCAB_SIZE = 9
TOKEN_NAMES = ('global', 'literal_false', 'literal_true', 'or_unsatisfied',
               'or_satisfied', 'xor_even_unsatisfied', 'xor_even_satisfied',
               'xor_odd_unsatisfied', 'xor_odd_satisfied')


def normalize_constraint(constraint):
    """Return (operator, literals, rhs); XOR rhs applies to signed input literals."""
    op = constraint['op']
    if op not in ('or', 'xor'):
        raise ValueError('Operator must be or or xor')
    literals = list(constraint['literals'])
    if any(not isinstance(v, int) or isinstance(v, bool) or v == 0 for v in literals):
        raise ValueError('Literals must be nonzero integer variable IDs')
    if op == 'or':
        if constraint.get('rhs') not in (None, 1):
            raise ValueError('OR clauses must target true')
        return OR, sorted(set(literals), key=lambda v: (abs(v), v)), 1
    rhs = constraint.get('rhs')
    rhs = 1 if rhs is None else rhs
    if rhs not in (0, 1):
        raise ValueError('XOR rhs must be 0 or 1')
    support = set()
    for literal in literals:
        rhs ^= int(literal < 0)
        variable = abs(literal)
        if variable in support:
            support.remove(variable)
        else:
            support.add(variable)
    return XOR, sorted(support), rhs


def pack_constraints(formulas, max_clauses=None):
    if not formulas or any(not f for f in formulas):
        raise ValueError('Expected a nonempty batch of nonempty constraint lists')
    maximum = max(map(len, formulas)) if max_clauses is None else max_clauses
    if maximum < max(map(len, formulas)):
        raise ValueError('max_clauses is too small')
    normalized = [[normalize_constraint(c) for c in f] for f in formulas]
    width = max(1, max(len(c[1]) for f in normalized for c in f))
    literals = torch.zeros((len(formulas), maximum, width), dtype=torch.long)
    operators = torch.empty((len(formulas), maximum), dtype=torch.long)
    rhs = torch.empty_like(operators)
    for b, formula in enumerate(normalized):
        for j in range(maximum):
            op, clause, target = formula[j % len(formula)]
            operators[b, j], rhs[b, j] = op, target
            if clause:
                literals[b, j, :len(clause)] = torch.tensor(clause)
    return literals, operators, rhs


def constraint_states(literals, operators, rhs, assignment):
    """Evaluate each constraint on assignment[B, n+1], with dummy variable 0."""
    if literals.abs().max() >= assignment.shape[-1]:
        raise ValueError('Literal outside assignment')
    values = assignment.bool().gather(1, literals.abs().flatten(1)).view_as(literals)
    values = torch.where(literals > 0, values, ~values) & literals.ne(0)
    disjunction = values.any(-1)
    parity = values.long().sum(-1).remainder(2)
    return torch.where(operators == OR, disjunction, parity == rhs)


def encode_inputs(literals, operators, rhs, assignment):
    """Encode supplied candidate values, without consulting supervision labels."""
    if assignment.ndim != 2 or not torch.all((assignment == 0) | (assignment == 1)):
        raise ValueError('Assignment must be a batch of Boolean values')
    state = constraint_states(literals, operators, rhs, assignment)
    base = torch.where(operators == OR, 3, 5 + 2 * rhs)
    return torch.cat((torch.zeros((len(assignment), 1), dtype=torch.long, device=assignment.device),
                      expand_not(assignment[:, 1:].bool()).long() + 1,
                      base + state.long()), dim=1)


def decode_assignment(logits, num_variables):
    """Decode only Boolean classes at the reference branch's odd literal nodes."""
    bits = logits[:, 1:2*num_variables+1:2, 1:3].argmax(-1).bool()
    return torch.cat((torch.zeros((len(bits), 1), dtype=torch.bool, device=bits.device), bits), dim=1)


def refresh_inputs(logits, literals, operators, rhs, num_variables):
    """Optionally rebuild a candidate input; never predict operator/parity tokens."""
    return encode_inputs(literals, operators, rhs, decode_assignment(logits, num_variables))


def collate_fn(batch, num_variables, max_clauses=None, augment=False, generator=None,
               materialize_mask=True, max_mask_bytes=2**30):
    if augment:
        raise NotImplementedError('Mixed SAT augmentation has not been implemented')
    literals, operators, rhs = pack_constraints([row['constraints'] for row in batch], max_clauses)
    if literals.abs().max() > num_variables:
        raise ValueError('Literal exceeds num_variables')
    answers = torch.tensor([[0] + list(row['assignment']) for row in batch], dtype=torch.long)
    if answers.shape != (len(batch), num_variables+1) or not torch.all((answers == 0) | (answers == 1)):
        raise ValueError('Invalid assignment labels')
    if not constraint_states(literals, operators, rhs, answers).all():
        raise ValueError('Assignment does not satisfy mixed OR/XOR constraints')
    initial = torch.cat((torch.zeros((len(batch), 1), dtype=torch.long),
                         torch.randint(0, 2, (len(batch), num_variables), generator=generator)), dim=1)
    x = encode_inputs(literals, operators, rhs, initial)
    y = encode_inputs(literals, operators, rhs, answers)
    y_mask = torch.zeros_like(y)
    y_mask[:, :2*num_variables+1] = 1
    spec = AttentionMaskSpec(literals, num_variables)
    return dict(x=x, y=y, y_mask=y_mask, clauses=literals, operators=operators, rhs=rhs,
                n=num_variables, attn_mask=spec.materialize(max_mask_bytes) if materialize_mask else spec)


def from_cnf_xor(or_clauses, xor_equations):
    """Adapt the exact native-XOR export made by compress_cnf.extract_xors."""
    return ([dict(op='or', literals=list(c)) for c in or_clauses] +
            [dict(op='xor', literals=list(support), rhs=int(rhs)) for support, rhs in xor_equations])


def create_dataloader(split, batch_size, rank, world_size, dataset_name, num_variables,
                      max_clauses, augment=False, repeat=1, num_workers=1, seed=42,
                      materialize_mask=True, max_mask_bytes=2**30):
    """Local JSONL or Hugging Face rows with constraints and assignment fields."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader, DistributedSampler
    path = Path(dataset_name)
    if path.is_dir() and (path / f'{split}.jsonl').exists():
        dataset = load_dataset('json', data_files={split: str(path / f'{split}.jsonl')})[split]
    elif path.is_file() and path.suffix in ('.json', '.jsonl'):
        dataset = load_dataset('json', data_files={split: str(path)})[split]
    else:
        dataset = load_dataset(dataset_name)[split]
    is_train = 'train' in split
    dataset = dataset.repeat(repeat if is_train else 1)
    return DataLoader(dataset, batch_size=batch_size,
        collate_fn=partial(collate_fn, num_variables=num_variables, max_clauses=max_clauses,
                           augment=augment and is_train, materialize_mask=materialize_mask,
                           max_mask_bytes=max_mask_bytes),
        sampler=DistributedSampler(dataset, rank=rank, num_replicas=world_size,
                                   shuffle=is_train, drop_last=True, seed=seed),
        drop_last=True, pin_memory=True, persistent_workers=num_workers > 0,
        num_workers=num_workers, prefetch_factor=2 if num_workers else None), {
        'vocab_size': VOCAB_SIZE, 'seq_len': 2*num_variables+max_clauses+1, 'is_causal': False}
