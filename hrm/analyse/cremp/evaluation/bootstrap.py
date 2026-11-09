"""Import the immutable A implementation without copying or editing its source."""
from hrm.paths import analysis_directory
from pathlib import Path
import sys
SOURCE = analysis_directory(__file__).parents[1] / 'releases/pilot_v1'
if not (SOURCE / 'release_manifest.json').exists():
    raise RuntimeError(f'Missing frozen A source: {SOURCE}')
pass
