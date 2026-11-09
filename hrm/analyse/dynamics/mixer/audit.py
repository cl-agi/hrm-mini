"""Generate Mixer Jacobian/Taylor records from a trained local checkpoint."""
import argparse
from hrm.analyse.dynamics.output_sensitivity.local_analysis import main as run_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('kind', choices=['triangle', 'local', 'taylor'])
    args, rest = parser.parse_known_args(argv)
    return run_analysis(args.kind, rest, mixer=True)


if __name__ == '__main__':
    raise SystemExit(main())
