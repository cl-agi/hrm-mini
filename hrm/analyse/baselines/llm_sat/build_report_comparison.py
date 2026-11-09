"""Join the current 32 settings to the preserved original report values."""
from hrm.paths import analysis_directory
import csv
from pathlib import Path

DATA = analysis_directory(__file__)
HISTORICAL = DATA.parent / 'llm_reference/report_comparison.csv'


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    historical = {}
    for row in read_csv(HISTORICAL):
        key = (row['model'], row['task'], row['scale'])
        value = (row['old_model'], float(row['old_report_percent']))
        assert key not in historical or historical[key] == value
        historical[key] = value
    current = read_csv(DATA / 'current_settings.csv')
    assert len(current) == len(historical) == 32
    rows = []
    for row in current:
        key = (row['model'], row['task'], row['scale'])
        old_model, old_score = historical[key]
        new_score = 100 * int(row['n_correct']) / int(row['n_trials'])
        rows.append({**{name: row[name] for name in ('model', 'task', 'scale', 'n_trials',
                                                   'n_correct', 'n_unique_problems')},
                     'old_model': old_model, 'old_report_percent': old_score,
                     'new_percent': new_score,
                     'difference_percentage_points': round(new_score - old_score, 10)})
    with (DATA / 'report_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} comparisons; original report values preserved.')


if __name__ == '__main__':
    main()
