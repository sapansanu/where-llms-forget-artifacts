"""Run the case study: every example under every condition (eviction method x budget x query knowledge, plus the full
cache), logging the answer, its correctness, and how much of each answer-bearing item survived eviction.

    uv run python -m memloss.run --task A --n 2 --model Qwen/Qwen2.5-0.5B-Instruct --out results/smoke.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import torch
from kvpress import SnapKVPress, StreamingLLMPress, TOVAPress
from transformers import pipeline

from memloss import retention, scoring, tasks

PRESSES = {
    "snapkv": lambda r: SnapKVPress(compression_ratio=r, window_size=64),
    "tova": lambda r: TOVAPress(compression_ratio=r),
    "streaming": lambda r: StreamingLLMPress(compression_ratio=r, n_sink=4),
}
BUDGETS = (0.75, 0.9)  # fraction evicted: keep 25% and 10% of the cache


def conditions(budgets=BUDGETS) -> list[dict]:
    out = [{"press": "full", "ratio": 0.0, "qk": qk} for qk in ("agnostic", "aware")]
    out += [{"press": p, "ratio": r, "qk": qk} for p in PRESSES for r in budgets for qk in ("agnostic", "aware")]
    return out


def build_examples(task: str, n: int, seed: int, n_records: int = 200) -> list[tasks.Example]:
    if task == "A":
        return tasks.synthetic_set(n_per_cell=n, n_records=n_records, seed=seed)
    from datasets import load_dataset
    rows = list(load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"))
    return tasks.hotpot_set(rows, n_questions=n, seed=seed)


def context_for(ex: tasks.Example, qk: str) -> str:
    # Query-aware: the question is also in the context, so the compressor sees it (SnapKV's window, TOVA's last token).
    return ex.context if qk == "agnostic" else f"{ex.context}\n\nQuestion: {ex.question}"


def score(ex: tasks.Example, answer: str) -> dict:
    if ex.task == "A":
        return {"correct": scoring.code_correct(answer, ex.answers[0])}
    em = max(scoring.exact_match(answer, g) for g in ex.answers)
    return {"correct": em, "f1": max(scoring.f1(answer, g) for g in ex.answers)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["A", "B"], required=True)
    ap.add_argument("--n", type=int, required=True, help="examples per cell (A) or questions (B)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--conditions", default="all", help="comma list of press names to run, or 'all'")
    ap.add_argument("--n-records", type=int, default=200, help="Task A load: records in the context")
    ap.add_argument("--qk", default="agnostic,aware", help="comma list of query-knowledge settings to run")
    ap.add_argument("--protect", action="store_true",
                    help="keep the answer-bearing item in the cache at the same budget (compressed conditions only)")
    ap.add_argument("--sham", action="store_true",
                    help="control for --protect: keep a random non-answer span of the same token length instead")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = pipeline("kv-press-text-generation", model=args.model, device=device, torch_dtype=dtype,
                    model_kwargs={"attn_implementation": "sdpa"})
    examples = build_examples(args.task, args.n, args.seed, args.n_records)[args.shard :: args.n_shards]
    conds = [c for c in conditions() if (args.conditions == "all" or c["press"] in args.conditions.split(","))
             and c["qk"] in args.qk.split(",")]
    if args.protect or args.sham:
        conds = [c for c in conds if c["press"] in PRESSES]
    if "closed" in args.conditions.split(","):  # opt-in control: the question with no context (parametric knowledge)
        conds.append({"press": "closed", "ratio": 0.0, "qk": "agnostic"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["id"], r["press"], r["ratio"], r["qk"]) for r in map(json.loads, out.open())}  # one mode per file
    rec = retention.RetentionRecorder()
    asked_closed = set()
    started, n_new = time.time(), 0
    with rec, out.open("a") as f:
        for ex in examples:
            positions = {qk: retention.span_token_positions(pipe.tokenizer, context_for(ex, qk), ex.spans)
                         for qk in ("agnostic", "aware")}
            support = {qk: retention.span_token_positions(pipe.tokenizer, context_for(ex, qk),
                                                          ex.meta.get("support_spans", []))
                       for qk in ("agnostic", "aware")}
            protect = {qk: sorted({i for p in retention.span_token_positions(
                pipe.tokenizer, context_for(ex, qk), ex.meta["protect_spans"]) for i in p})
                for qk in ("agnostic", "aware")} if (args.protect or args.sham) else None
            if args.sham:  # same number of tokens, contiguous, inside the context proper, disjoint from the item
                protect = {qk: retention.sham_positions(
                    protect[qk], retention.span_token_positions(pipe.tokenizer, context_for(ex, qk),
                                                                [(0, len(ex.context))])[0], seed_key=ex.id)
                    for qk in protect}
            key = {qk: retention.span_token_positions(pipe.tokenizer, context_for(ex, qk), [ex.meta["key_span"]])[0]
                   for qk in ("agnostic", "aware")} if "key_span" in ex.meta else None
            for c in conds:
                if (ex.id, c["press"], c["ratio"], c["qk"]) in done:
                    continue
                if c["press"] == "closed":  # no context, so one run per question is enough
                    if ex.question in asked_closed:
                        continue
                    asked_closed.add(ex.question)
                press = None if c["press"] in ("full", "closed") else PRESSES[c["press"]](c["ratio"])
                rec.reset(protect[c["qk"]] if protect else None)
                context = "" if c["press"] == "closed" else context_for(ex, c["qk"])
                answer = pipe(context, question=ex.question, press=press,
                              max_new_tokens=args.max_new_tokens)["answer"]
                full = 0.0 if c["press"] == "closed" else 1.0  # closed book: the item is not in the context at all
                kept = [rec.retention(p) for p in positions[c["qk"]]] if press else [full] * len(ex.spans)
                cover = [rec.coverage(p) for p in positions[c["qk"]]] if press else \
                    [{"mean": full, "full_heads": full, "any_head_layers": full}] * len(ex.spans)
                row = {"id": ex.id, "task": ex.task, **ex.meta, **c, "model": args.model, "answer": answer,
                       **score(ex, answer), "retention": kept,
                       "retention_min": min(kept) if kept and None not in kept else None,
                       "retention_mean": sum(kept) / len(kept) if kept and None not in kept else None,
                       "coverage": cover,
                       "full_heads_min": min(c_["full_heads"] for c_ in cover) if cover and None not in cover else None,
                       "any_head_layers_min": min(c_["any_head_layers"] for c_ in cover) if cover and None not in cover else None,
                       "span_tokens": [len(p) for p in positions[c["qk"]]]}
                if support[c["qk"]]:  # Task B: retention of the whole supporting sentences (secondary measure)
                    row["support_retention"] = [rec.retention(p) for p in support[c["qk"]]] if press \
                        else [full] * len(support[c["qk"]])
                if key:  # Task A: the record's key (its address) must survive too, not just the value
                    row["key_retention_min"] = rec.retention(key[c["qk"]]) if press else full
                    row["key_full_heads"] = (rec.coverage(key[c["qk"]]) or {}).get("full_heads") if press else full
                    row["answer_source"] = tasks.code_source(ex, answer)
                row["protected"] = bool(args.protect)
                row["sham"] = bool(args.sham)
                row.pop("key_span", None)
                row.pop("protect_spans", None)
                row.pop("support_spans", None)
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_new += 1
    print(f"wrote {n_new} rows to {out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
