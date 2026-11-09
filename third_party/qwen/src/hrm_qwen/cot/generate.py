"""Distil Sudoku reasoning traces from a strong API model, keeping only the verified-correct ones.

The reasoning baselines so far had no reference reasoning: `llm_reason` trained on the model's
*own* traces with the answer as the only target, and `llm_rlvr` shaped the trace by outcome
alone. Neither got a 4B model past ~35 blanks. This script buys reference reasoning instead:
a large reasoning model (DeepSeek V4 Pro through OpenRouter) is asked the same prompt the
student sees, its thinking is captured, and the trace is kept **only if the 81-digit answer
matches the gold solution exactly**. The student is then SFT'd on `[prompt | <think> trace
</think> | answer]` with the loss on the whole completion (`train.py`), i.e. it learns to
reason the way the teacher does, not merely to read off an answer.

Data hygiene: puzzles come from the 1000-row `sudoku-extreme-1k` training split only -- the
same puzzles every other arm trains on -- optionally under the same band/stack/digit
augmentation (`--augment`, addressed by `(seed, index)` like every other baseline). Nothing
from `test_hard` is ever sent to the API.

Resumable: each record is appended to `--out` as soon as it is graded, and indices already
present are skipped on rerun. Wrong or unparseable answers are logged too (with `exact=false`)
so yield and cost can be reported; `train.py` filters on `exact`.

    python -m hrm_qwen.cot.generate --count 1000 --out exp/sudoku/records/qwen/teacher/train.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time

import requests

from hrm_qwen.cot.scoring import grade, load_puzzles, make_prompt, puzzle_with_empties, augmented_rows
from hrm_qwen.cot.generation_config import prepare_run, load_seed, trace_id, ratings_for, execution_lock, read_records

# Two backends for the same model. OpenRouter routes across third-party hosts (some cap
# reasoning at 32k, and a key has a daily spend limit); DeepSeek's own API is one host with much
# higher concurrency and no per-day cap, and returns the thinking as `reasoning_content`.
# Models whose thinking comes back summarised or encrypted (Claude, GPT) are instead asked to write
# the whole derivation as visible output; the student trains on that text as its reasoning.
VISIBLE_SUFFIX = ("\n\nShow your complete working in your answer: write out every deduction, candidate "
                  "elimination and any trial-and-error you need, step by step, and only then give the final grid.")

BACKENDS = {
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions", "key": "OPENROUTER_API_KEY",
                   "model": "deepseek/deepseek-v4-pro"},
    "deepseek": {"url": "https://api.deepseek.com/chat/completions", "key": "DEEPSEEK_API_KEY",
                 "model": "deepseek-v4-pro"},
}


def api_key(var: str) -> str:
    key = os.environ.get(var)
    if not key:
        sys.exit(f"{var} not set in environment")
    return key


def ask(backend: str, key: str, model: str, prompt: list[dict], max_tokens: int, reasoning_tokens: int,
        temperature: float, timeout: float, providers: list[str] | None, retries: int = 4,
        visible: bool = False, continuations: int = 0, _add_suffix: bool = True) -> dict:
    """One chat completion with thinking on. `providers` pins OpenRouter's routing: several of the
    hosts serving DeepSeek V4 Pro silently cap reasoning at 32,768 tokens or drop long
    generations, and an Extreme puzzle needs ~40k+ tokens of thinking."""
    if visible and _add_suffix:
        prompt = [{"role": "user", "content": prompt[0]["content"] + VISIBLE_SUFFIX}]
    body = {"model": model, "messages": prompt, "max_tokens": max_tokens, "temperature": temperature}
    if backend == "openrouter":
        body["reasoning"] = {"enabled": False} if visible else {"enabled": True, "max_tokens": reasoning_tokens}
        if providers:
            body["provider"] = {"order": providers, "allow_fallbacks": False}
    else:
        body["thinking"] = {"type": "disabled" if visible else "enabled"}
    delay, attempt = 5.0, 0
    while True:
        try:
            r = requests.post(BACKENDS[backend]["url"], headers={"Authorization": f"Bearer {key}"}, json=body,
                              timeout=timeout)
            j = r.json() if r.status_code == 200 else {}
            if "choices" in j and j["choices"][0].get("finish_reason") != "error":
                # Visible-mode output cut at the provider's cap: hand the partial answer back as the
                # previous assistant turn and ask for the continuation (Claude 5 rejects prefill).
                if (visible and continuations > 0 and j["choices"][0].get("finish_reason") == "length"
                        and backend == "openrouter"):
                    partial = j["choices"][0]["message"].get("content") or ""
                    follow = [{"role": "assistant", "content": partial},
                              {"role": "user", "content": "Your answer was cut off by the output limit. Continue "
                               "exactly from where you stopped, without repeating anything, and finish with the "
                               "final grid as nine lines of nine digits."}]
                    more = ask(backend, key, model, body["messages"] + follow, max_tokens, reasoning_tokens,
                               temperature, timeout, providers, retries, visible=True,
                               continuations=continuations - 1, _add_suffix=False)
                    if "choices" in more:
                        m2 = more["choices"][0]["message"]
                        m2["content"] = partial + (m2.get("content") or "")
                        u1, u2 = j.get("usage", {}), more.get("usage", {})
                        more["usage"] = {**u2, "completion_tokens": u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0),
                                         "cost": (u1.get("cost") or 0) + (u2.get("cost") or 0), "continued": True}
                        more["choices"][0]["finish_reason"] = more["choices"][0].get("finish_reason")
                        return more
                return j
            err = j.get("error") or r.text[:300] or f"finish_reason=error from {j.get('provider')}"
            err = str(err)
            # Rate limits consume the same bounded transport retry budget.
            if r.status_code == 429:
                err = "HTTP 429: " + err
        except (requests.RequestException, ValueError, IndexError, KeyError) as e:  # noqa: PERF203
            err = repr(e)
        attempt += 1
        if attempt >= retries:
            return {"error": err}
        time.sleep(delay)
        delay *= 2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="deepseek", choices=list(BACKENDS))
    p.add_argument("--model", default=None, help="default: the backend's DeepSeek V4 Pro id")
    p.add_argument("--data-dir", default=None, help="local CSV override; default pinned HF train split")
    p.add_argument("--offset", type=int, default=0, help="first puzzle / augmentation index")
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--select-count", type=int, default=None, help="freeze at most this many eligible puzzles after exclusions and ordering")
    p.add_argument("--augment", action="store_true", help="augmented views (index addresses the view)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--execute", action="store_true", help="make paid API calls after preparation")
    p.add_argument("--layout", default="rows", choices=["rows", "line"])
    p.add_argument("--max-tokens", type=int, default=131072, help="reasoning + answer budget at the API")
    p.add_argument("--reasoning-tokens", type=int, default=120000, help="reasoning budget at the API")
    p.add_argument("--providers", default="novita,parasail",
                   help="comma-separated OpenRouter provider order, no fallbacks; '' = let OpenRouter route")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=7200, help="seconds per request")
    p.add_argument("--workers", type=int, default=4,
                   help="maximum concurrent teacher requests")
    p.add_argument("--out", default="exp/sudoku/records/qwen/teacher/train.jsonl", help="JSONL; adjacent configuration is frozen before execution")
    p.add_argument("--retry-errors", action="store_true",
                   help="re-attempt indices whose record is an API error (rate limit, timeout)")
    p.add_argument("--visible-cot", action="store_true",
                   help="thinking off; ask for the full derivation as output and store it as the trace")
    p.add_argument("--continuations", type=int, default=0,
                   help="visible mode: continue an output cut at the cap this many times (prefill)")
    p.add_argument("--max-cost", type=float, default=None, help="stop starting new requests past this spend ($)")
    p.add_argument("--order", default="shuffle", choices=["shuffle", "rating_asc", "rating_desc"])
    p.add_argument("--only-unsolved-in", default=None,
                   help="JSONL(s), comma-separated: skip puzzles that already have a correct trace there")
    p.add_argument("--min-rating", type=int, default=None, help="only puzzles with dataset rating >= this")
    p.add_argument("--empties", type=int, default=None,
                   help="reveal cells from the solution until this many blanks remain (easier variant of the same puzzle)")
    p.add_argument("--retry-wrong", action="store_true",
                   help="re-attempt indices that have no correct record yet (T=1 sampling gives a fresh trace)")
    args = p.parse_args()

    load_seed(args)
    args.model = args.model or BACKENDS[args.backend]["model"]
    rows = load_puzzles(args.data_dir, "train")
    if args.augment:
        puzzles = augmented_rows(rows, range(args.offset, args.offset + args.count), args.seed)
    else:
        puzzles = [(i, *rows[i]) for i in range(args.offset, min(args.offset + args.count, len(rows)))]

    if len(puzzles) != args.count:
        raise ValueError("Requested cohort exceeds the training split")
    prepare_run(args, puzzles)
    if not args.execute:
        return
    key = api_key(BACKENDS[args.backend]["key"])
    with execution_lock(args.out):
        cohort = prepare_run(args, puzzles)
        selected = set(cohort["selected_indices"])
        puzzles = [item for item in puzzles if item[0] in selected]
        done, solved, attempts = set(), set(), {}
        if os.path.exists(args.out):
            for rec in read_records(args.out):
                attempts[rec.get("source_index", rec["index"])] = attempts.get(rec.get("source_index", rec["index"]), 0) + 1
                if rec.get("exact"):
                    solved.add(rec.get("source_index", rec["index"]))
                skip = not (args.retry_errors and "error" in rec) and not (args.retry_wrong and not rec.get("exact"))
                if skip:
                    done.add(rec.get("source_index", rec["index"]))
        done |= solved  # a solved puzzle is never re-attempted
        if args.only_unsolved_in:
            for path in args.only_unsolved_in.split(","):
                if os.path.exists(path):
                    for rec in read_records(path):
                        if rec.get("exact"):
                            done.add(rec.get("source_index", rec["index"]))
        if args.min_rating is not None:
            import csv

            ratings = ratings_for(args.data_dir)
            done |= {i for i in range(len(ratings)) if ratings[i] < args.min_rating}
        todo = [(i, q, a) for i, q, a in puzzles if i not in done]
        if args.empties is not None:
            todo = [(i, puzzle_with_empties(q, a, args.empties, args.seed, i), a) for i, q, a in todo]
        order_index = {index: position for position, index in enumerate(cohort["selected_indices"])}
        todo.sort(key=lambda item: order_index[item[0]])
        print(f"[generate] {args.model}: {len(todo)} puzzles to do, {len(done)} already in {args.out}", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        lock = threading.Lock()
        previous_cost = sum(float((r.get("usage") or {}).get("cost") or 0)
                            for r in read_records(args.out)) if os.path.exists(args.out) else 0.0
        stats = {"n": 0, "exact": 0, "errors": 0, "tokens": 0, "cost": previous_cost}

        def work(item):
            i, q, a = item
            with lock:
                if args.max_cost is not None and stats["cost"] >= args.max_cost:
                    return  # budget spent; leave the puzzle for another run
            t0 = time.time()
            r = ask(args.backend, key, args.model, make_prompt(q, args.layout), args.max_tokens, args.reasoning_tokens,
                    args.temperature, args.timeout, [x for x in args.providers.split(",") if x], visible=args.visible_cot,
                    continuations=args.continuations)
            rec = {"index": i, "source_index": i, "variant_id": trace_id(q, a), "question": q, "answer": a, "augmented": args.augment, "seed": args.seed,
                   "backend": args.backend, "model": args.model, "layout": args.layout, "visible_cot": args.visible_cot,
                   "empties": args.empties, "blanks": q.count("0"),
                   "attempt": attempts.get(i, 0) + 1, "secs": round(time.time() - t0, 1)}
            if "error" in r:
                rec.update(error=r["error"], exact=False)
            else:
                msg = r["choices"][0]["message"]
                reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
                content = msg.get("content") or ""
                if args.visible_cot:
                    # the derivation *is* the output; grade its tail, and file the text as the trace
                    g = grade(f"<think>\n{content}\n</think>\n\n{content[-3000:]}", q, a)
                    reasoning, content = content, ("\n".join(a[k:k + 9] for k in range(0, 81, 9)) if g["exact"] else content[-3000:])
                else:
                    g = grade(f"<think>\n{reasoning}\n</think>\n\n{content}", q, a)
                rec.update(reasoning=reasoning, content=content, finish=r["choices"][0].get("finish_reason"),
                           provider=r.get("provider"), response_model=r.get("model"), usage=r.get("usage", {}), **g)
            with lock:
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                stats["n"] += 1
                stats["exact"] += int(rec.get("exact", False))
                stats["errors"] += int("error" in rec)
                u = rec.get("usage") or {}
                stats["tokens"] += u.get("completion_tokens", 0)
                stats["cost"] += float(u.get("cost", 0) or 0)
                if stats["n"] % 10 == 0 or stats["n"] == len(todo):
                    print(f"[generate] {stats['n']}/{len(todo)} done, {stats['exact']} correct "
                          f"({stats['exact'] / stats['n']:.0%}), {stats['errors']} errors, "
                          f"{stats['tokens'] / max(stats['n'] - stats['errors'], 1):.0f} completion tok/puzzle, "
                          f"${stats['cost']:.2f}", flush=True)

        with cf.ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(work, todo))



if __name__ == "__main__":
    main()
