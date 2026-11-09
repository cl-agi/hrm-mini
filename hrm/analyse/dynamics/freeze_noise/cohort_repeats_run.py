"""Compatibility entry for the shared freeze/noise controller."""
import sys
from .run import main
if __name__ == '__main__':
    sys.argv[1:1] = ['--protocol', 'cohort_repeats']
    main()
