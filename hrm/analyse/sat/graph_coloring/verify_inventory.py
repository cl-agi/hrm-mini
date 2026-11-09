"""Check the stored experiment inventory without importing ML dependencies."""
from hrm.paths import analysis_directory
from pathlib import Path
import hashlib
import json

P = analysis_directory(__file__)
root = P.parent.parent
manifest = json.loads((P/'inventory.json').read_text())
errors = []
for record in manifest['files']:
    path = root/record['path']
    if not path.is_file():
        errors.append(f"Missing: {record['path']}")
        continue
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    if path.stat().st_size != record['bytes'] or digest.hexdigest() != record['sha256']:
        errors.append(f"Changed: {record['path']}")
if errors:
    raise SystemExit('\n'.join(errors))
print(f"Verified {len(manifest['files'])} files across {len(manifest['experiments'])} experiment directories.")
