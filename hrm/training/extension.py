"""Budget decisions use accuracy fractions, not percentages."""
def trend_decision(history, min_gain=0.0005, max_drop=0.002):
    if len(history) < 6:
        return {'extend': False, 'reason': 'fewer than six evaluations'}
    tail = history[-6:]
    gain = sum(tail[-3:]) / 3 - sum(tail[:3]) / 3
    drop = max(tail) - tail[-1]
    rising = gain >= min_gain and drop <= max_drop
    return {'extend': rising, 'recent_mean_gain': gain, 'drop_from_recent_best': drop,
            'min_gain': min_gain, 'max_drop': max_drop,
            'reason': 'recent improvement' if rising else 'no clear recent improvement'}
