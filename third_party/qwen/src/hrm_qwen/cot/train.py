"""SFT a thinking LLM on distilled, verified-correct Sudoku reasoning traces.

Each example is `[chat prompt | <think>\\n trace \\n</think>\\n\\n answer <|im_end|>]`, with the trace
from `generate.py` (a strong API model) and the loss on **every completion token**: reasoning
and answer alike. That is the difference from `llm_reason.train`, which supervised only the 82
answer tokens and left the reasoning to drift; here the student is taught how to reason.

Only records with `exact == true` are used, i.e. traces whose final grid matched the gold
solution -- a wrong trace is not a demonstration of anything. Sequences are variable-length,
so the per-device batch is one sequence and the global batch comes from gradient accumulation.
Full fine-tune, fp32 master weights, FSDP, gradient checkpointing.

    torchrun --nproc-per-node 8 -m hrm_qwen.cot.train --model Qwen/Qwen3.5-4B \\
        --traces exp/sudoku/records/qwen/teacher/train.jsonl --output-dir exp/sudoku/records/qwen/student
"""

from __future__ import annotations

import argparse
import json
import os
import random

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist

from hrm_qwen.cot.prompt import make_prompt
from hrm_qwen.cot.run_config import freeze_run


def patch_fused_ce(model) -> bool:
    """Give a model Liger's fused linear cross-entropy when Liger has no patch for its type.

    `use_liger_kernel` only covers the model types in Liger's table (Qwen3.5 is one; NemotronH is
    not), and without it the [T x vocab] fp32 logits of a 100k-token trace are ~50 GB. This swaps
    the class `forward` for one that runs the backbone and feeds the last hidden states straight
    into the fused loss -- what Liger's own per-model patches do. Patched at class level, before
    FSDP wraps the model, so `lm_head` is unsharded when the loss reads its weight.
    """
    from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
    from transformers.modeling_outputs import CausalLMOutputWithPast

    if model.config.model_type in MODEL_TYPE_TO_APPLY_LIGER_FN:
        return False
    cls = type(model)
    original = cls.forward

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            return original(self, input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        n_items = kwargs.pop("num_items_in_batch", None)
        out = self.get_decoder()(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **kwargs)
        loss = LigerForCausalLMLoss(out[0], self.lm_head.weight, labels, self.config.hidden_size,
                                    num_items_in_batch=n_items)
        return CausalLMOutputWithPast(loss=loss)

    cls.forward = forward
    return True


def mamba_max_tokens(config) -> int | None:
    """Longest sequence a Mamba2 hybrid can train on before its kernels overflow int32 indexing.

    `mamba_split_conv1d_scan_combined` indexes the [tokens x in_proj width] activation with int32,
    so tokens x width must stay below 2^31: 122,685 tokens for Nemotron-3-Nano-4B (width 17,504).
    Past it the Triton kernel reads out of bounds -- a crash on one GPU, silent NaN gradients
    under FSDP. None for models without Mamba layers.
    """
    heads = getattr(config, "mamba_num_heads", None)
    if not heads:
        return None
    width = 2 * heads * config.mamba_head_dim + 2 * config.n_groups * config.ssm_state_size + heads
    return (2**31 - 1) // width


def build_example(tok, rec: dict) -> list[int]:
    """Prompt ids from the model's own template (which opens `<think>\\n`), then trace, close, answer."""
    prompt = tok.apply_chat_template(make_prompt(rec["question"], rec.get("layout", "rows")),
                                     tokenize=False, add_generation_prompt=True, enable_thinking=True)
    completion = rec["reasoning"].strip() + "\n</think>\n\n" + rec["content"].strip()
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]
    return p_ids, c_ids


def load_examples(tok, paths: list[str], max_len: int, epochs: int) -> tuple[list[dict], dict]:
    by_index = {}  # a puzzle may have several records (retries, repairs); a correct one wins
    for path in paths:
        with open(path) as stream:
            for line in stream:
                r = json.loads(line)
                if not by_index.get(r["index"], {}).get("exact"):
                    by_index[r["index"]] = r
    recs = list(by_index.values())
    kept, dropped_wrong, dropped_long, lengths = [], 0, 0, []
    for r in recs:
        if not r.get("exact"):
            dropped_wrong += 1
            continue
        p_ids, c_ids = build_example(tok, r)
        if len(p_ids) + len(c_ids) > max_len:
            dropped_long += 1
            continue
        lengths.append(len(p_ids) + len(c_ids))
        kept.append({"input_ids": p_ids + c_ids, "n_prompt": len(p_ids)})
    stats = {"n_records": len(recs), "n_correct": len(recs) - dropped_wrong, "n_used": len(kept),
             "n_repaired": sum(1 for r in recs if r.get("exact") and r.get("repaired")),
             "dropped_wrong": dropped_wrong, "dropped_long": dropped_long,
             "len_mean": sum(lengths) / max(len(lengths), 1), "len_max": max(lengths, default=0)}
    return kept * epochs, stats


def collate(features: list[dict]) -> dict[str, torch.Tensor]:
    assert len(features) == 1, "one variable-length sequence per device; use --grad-accum"
    f = features[0]
    ids = torch.tensor([f["input_ids"]], dtype=torch.long)
    labels = ids.clone()
    labels[:, : f["n_prompt"]] = -100  # loss on the completion (reasoning + answer) only
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--traces", required=True, nargs="+", help="JSONL files from generate.py / repair.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-len", type=int, default=131_072, help="drop longer sequences")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=None, help="frozen Unix-second seed by default")
    p.add_argument("--prepare-only", action="store_true", help="freeze configuration before launching distributed training")
    p.add_argument("--grad-accum", type=int, default=4, help="global batch = world x this")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--no-liger", action="store_true")
    p.add_argument("--max-steps", type=int, default=-1, help="smoke tests")
    p.add_argument("--no-save", action="store_true", help="smoke tests")
    p.add_argument("--run-name", default=None)
    p.add_argument("--wandb-project", default=None, help="optional W&B project; disabled by default")
    args = p.parse_args()

    freeze_run(args)
    if args.prepare_only:
        return
    from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments
    from hrm_qwen.sft.eval import load_causal_lm

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    tok = AutoTokenizer.from_pretrained(args.model)
    limit = mamba_max_tokens(AutoConfig.from_pretrained(args.model))
    if limit is not None and args.max_len > limit:
        if rank == 0:
            print(f"[llm_cot] --max-len {args.max_len} -> {limit}: longer sequences overflow the Mamba2 kernels",
                  flush=True)
        args.max_len = limit
    examples, stats = load_examples(tok, args.traces, args.max_len, args.epochs)
    random.Random(args.seed).shuffle(examples)
    if not examples:
        raise SystemExit(f"no usable traces in {args.traces}: {stats}")

    model = load_causal_lm(args.model, torch.float32)
    model.config.use_cache = False
    fused_ce_patched = not args.no_liger and patch_fused_ce(model)
    present = {type(m).__name__ for m in model.modules()}
    layers = [n for n in (getattr(model, "_no_split_modules", None) or []) if n in present]

    steps = len(examples) // (world * args.grad_accum)
    if rank == 0:
        print(f"[llm_cot] {args.model} <- {' '.join(args.traces)}\n  traces  : {stats}\n"
              f"  steps   : {steps} at global batch {world * args.grad_accum}, lr {args.lr}, "
              f"{args.epochs} epoch(s)\n"
              f"  loss    : {'fused CE via patch_fused_ce' if fused_ce_patched else 'Liger' if not args.no_liger else 'unfused'}",
              flush=True)
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    run_name = args.run_name or os.path.basename(args.output_dir.rstrip("/"))

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            run_name=run_name,
            seed=args.seed,
            num_train_epochs=1,  # epochs are baked into `examples`
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=args.warmup_steps,
            weight_decay=0.0,
            adam_beta2=0.95,
            max_grad_norm=1.0,
            bf16=True,
            gradient_checkpointing=True,
            # Liger's fused linear cross-entropy: the [T x 248k] logits of a 100k-token trace
            # would be ~100 GB in fp32; the fused kernel never materialises them.
            use_liger_kernel=not args.no_liger and not fused_ce_patched,
            logging_steps=args.logging_steps,
            save_strategy="no",
            report_to="wandb" if rank == 0 and args.wandb_project else "none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            fsdp="full_shard auto_wrap",
            fsdp_config={"version": 2, "transformer_layer_cls_to_wrap": layers,
                         "state_dict_type": "FULL_STATE_DICT", "reshard_after_forward": True},
        ),
        train_dataset=examples,
        data_collator=collate,
    )
    if rank == 0 and args.wandb_project:
        import wandb

        wandb.init(project=args.wandb_project, name=run_name,
                   config={"base": args.model, "traces": " ".join(args.traces), **stats, "epochs": args.epochs,
                           "lr": args.lr, "global_batch": world * args.grad_accum, "loss": "full_completion"},
                   settings=wandb.Settings(x_disable_stats=True))
    trainer.train()
    if args.no_save:
        return
    trainer.save_model(os.path.join(args.output_dir, "final"))
    if rank == 0:
        tok.save_pretrained(os.path.join(args.output_dir, "final"))
        with open(os.path.join(args.output_dir, "final", "train_meta.json"), "w") as f:
            json.dump({"steps": trainer.state.global_step, "seed": args.seed, **stats}, f, indent=2)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
