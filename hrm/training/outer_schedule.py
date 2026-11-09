"""Epoch-based training horizon, independent of evaluation depth."""
def outer_for_epoch(config, epoch):
    schedule = config.outer_schedule
    if not schedule:
        return config.cycles_per_data
    if len(schedule) != config.epochs or any(v < 1 or v > config.cycles_per_data for v in schedule):
        raise ValueError("outer_schedule must have epochs entries within [1, cycles_per_data]")
    if any(a > b for a, b in zip(schedule, schedule[1:])):
        raise ValueError("outer_schedule must be nondecreasing for a persistent ACT pool")
    return schedule[epoch]
