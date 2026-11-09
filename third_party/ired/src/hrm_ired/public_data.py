"""Verified public inputs linked by the pinned IRED upstream download scripts."""
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from hrm import hub

SOURCES = {
    'sudoku': ('https://powei.tw/sudoku.zip',
               '0d1102e846e22563d5ecdfc0c635df489a2daaadae068a856cfb43b1a8fcb49a',
               'sudoku', ('features.pt', 'labels.pt')),
    'sudoku-rrn': ('https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1',
                   '99f5c1f3f9a7c26e2e52d087dba2b312a816c9a54c638811455db72b8d3aa30d',
                   'sudoku-hard', ('train.csv', 'valid.csv', 'test.csv')),
}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def directory(identifier, path=None):
    url, expected, prefix, names = SOURCES[identifier]
    if path is not None:
        root = Path(path).expanduser().resolve()
        for name in names:
            hub.local_file(root / name)
        return root
    root = hub.cache_path('sudoku', f'ired/{identifier}/{expected}/placeholder').parent
    manifest = root / 'source.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['archive_sha256'] != expected or any(digest(root / name) != saved['files'][name] for name in names):
            raise ValueError(f'Public IRED input cache changed: {root}')
        return root
    # Extract only named numerical inputs, never image tensors or arbitrary paths.
    with tempfile.TemporaryFile(dir=root) as archive:
        with urllib.request.urlopen(url, timeout=60) as response:
            shutil.copyfileobj(response, archive)
        archive.seek(0)
        if hashlib.file_digest(archive, 'sha256').hexdigest() != expected:
            raise ValueError(f'Upstream archive changed: {url}')
        archive.seek(0)
        with zipfile.ZipFile(archive) as z:
            for name in names:
                def write(target, member=f'{prefix}/{name}'):
                    with z.open(member) as source, target.open('wb') as destination:
                        shutil.copyfileobj(source, destination)
                hub._write(root / name, write)
    payload = dict(url=url, archive_sha256=expected, files={name: digest(root / name) for name in names})
    hub._write(manifest, lambda p: p.write_text(json.dumps(payload, indent=2)+'\n'))
    return root
