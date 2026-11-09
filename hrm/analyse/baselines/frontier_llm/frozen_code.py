"""Load the exact historical solver definitions without dataset or API side effects."""

import ast
from collections import deque
from functools import lru_cache, partial
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
SOURCE_FILES = {'original/openrouter.py': 'openrouter.py', 'runtime/transport.py': 'transport.py', 'runtime/retry.py': 'retry.py', 'runtime/engine.py': 'engine.py', 'runtime/audit_full.py': 'audit_full.py', 'protocol/maze_shortest_v003.py': 'maze_shortest_v003.py', 'output_contract/dataset/sat3.py': 'sat3.py', 'output_contract/dataset/maze.py': 'maze.py', 'output_contract/dataset/sudoku.py': 'sudoku.py', 'original/dataset/sat3.py': 'frozen_sources_original_dataset_sat3.py', 'original/dataset/sudoku.py': 'frozen_sources_original_dataset_sudoku.py'}

def source_path(relative):
    return ROOT / SOURCE_FILES[relative]


def verify_sources():
    from hrm.paths import repo_path
    manifest=json.loads(repo_path('SOURCE.json').read_text())['integrated_files']
    verified={}
    for relative,filename in SOURCE_FILES.items():
        path=ROOT/filename
        if not path.exists():continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        key=str(path.relative_to(repo_path('.')))
        if digest != manifest[key]:raise ValueError('Packaged source changed: '+filename)
        verified[relative]=digest
    return verified


def definitions(relative, names, namespace=None):
    """Compile only named, unchanged function ASTs; skip all module-level effects."""
    path = source_path(relative)
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError("Missing frozen definitions in " + relative)
    scope = {"json": json, "re": re, "deque": deque, **(namespace or {})}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


def _module(relative, name):
    spec = importlib.util.spec_from_file_location(name, source_path(relative))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_original():
    verify_sources()
    client = definitions("original/openrouter.py", {
        "robust_json_parse", "_extract_text_from_content", "_collect_text_values"})
    engine = definitions("runtime/engine.py", {
        "iter_sse", "build_payload", "context_admission"})
    client._iter_sse_events = engine.iter_sse
    formatters = {
        task: definitions("output_contract/dataset/" + filename, {"get_prompt", "format_prompt"})
        for task, filename in {"sudoku": "sudoku.py", "maze": "maze.py", "sat": "sat3.py"}.items()
    }
    maze = definitions("protocol/maze_shortest_v003.py", {"diagnose", "validate"})
    validators = {
        "sudoku": definitions("original/dataset/sudoku.py", {"validate"}).validate,
        "sat": definitions("original/dataset/sat3.py", {"validate"}).validate,
        "maze": partial(maze.validate, height=30, width=30),
    }
    return SimpleNamespace(
        client=client, engine=engine, formatters=formatters, validators=validators,
        maze_diagnose=partial(maze.diagnose, height=30, width=30),
        transport=_module("runtime/transport.py", "llm_release_original_transport"),
        retry=_module("runtime/retry.py", "llm_release_original_retry"),
    )


def original_grade(task, problem, final_content):
    original = load_original()
    result = original.transport.grade(
        original.client, task, problem, final_content, original.validators[task],
        maze_dimensions={"height": 30, "width": 30} if task == "maze" else None)
    if task == "maze" and "parsed_answer" in result:
        result["maze_diagnostics"] = original.maze_diagnose(problem["question"], result["parsed_answer"])
    return result


def robust_json_parse(content):
    return load_original().client.robust_json_parse(content)
