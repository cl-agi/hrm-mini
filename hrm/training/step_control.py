"""Optimizer-step budget with the existing small-data loader semantics."""
def validate_budget(max_steps, eval_interval_steps, cycles_per_data, evaluations):
    if min(max_steps, eval_interval_steps, cycles_per_data) <= 0:
        raise ValueError('Step budgets must be positive')
    if max_steps % eval_interval_steps or eval_interval_steps % cycles_per_data:
        raise ValueError('Evaluation boundaries must align with complete outer groups')
    if max_steps // eval_interval_steps != evaluations:
        raise ValueError('epochs is the legacy evaluation-stage count, not data passes')


def optimizer_batches(loader, max_steps, cycles_per_data):
    if len(loader) == 0:
        raise ValueError('Training loader has no complete batches')
    iterator = iter(loader)
    data_pass = 0
    for step in range(1, max_steps + 1):
        outer_index = (step - 1) % cycles_per_data
        if outer_index == 0:
            try:
                batch = next(iterator)
            except StopIteration:
                data_pass += 1
                # Deliberately match the existing small-data implementation:
                # same DistributedSampler epoch; augmentation workers continue.
                iterator = iter(loader)
                batch = next(iterator)
        yield step, outer_index, data_pass, batch


def evaluation_due(step, eval_interval_steps):
    return step % eval_interval_steps == 0
