"""Task 1 key similarity measured inside the reader, not only at the input embeddings.

For each example, every record's key span (e.g. "Heron-3541") is located in the templated context. Its representation
is the mean over the span's tokens of (a) the input embedding and (b) the pre-RoPE key vector k_proj(h) in every
layer, all KV heads concatenated. We report, per similarity tier, the mean cosine between the target's key and its
competitors' keys, per layer. Pre-RoPE keys are used so that position does not enter the comparison.

    python -m memloss.keysim --model Qwen/Qwen2.5-7B-Instruct --n 10 --out results/keysim/qwen.json
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memloss import retention, tasks

KEY_RE = re.compile(r"The access code for (\S+) is ")


def key_spans(context: str) -> list[tuple[int, int]]:
    """Character spans of every record's key in the context, in order."""
    return [m.span(1) for m in KEY_RE.finditer(context)]


def _cos_to_target(vecs: torch.Tensor, target: int, mean: torch.Tensor | None = None) -> torch.Tensor:
    """Cosine between the target row and every other row of `vecs` ([n, d]). If `mean` is given (the average key over
    all records of all examples and tiers) it is subtracted first, so a component every key shares, such as a large
    k_proj bias, does not dominate the cosine while differences between tiers remain."""
    v = vecs.float() - (mean if mean is not None else 0)
    v = torch.nn.functional.normalize(v, dim=-1)
    sims = v @ v[target]
    return torch.cat([sims[:target], sims[target + 1:]])


@torch.no_grad()
def pooled_keys(model, tokenizer, ex: tasks.Example) -> tuple[int, torch.Tensor, list[torch.Tensor]]:
    """(target index, input embeddings [n, d], pre-RoPE keys per layer [n, d_kv]) pooled over each key span."""
    spans = key_spans(ex.context)
    target = spans.index(tuple(ex.meta["key_span"]))
    positions = retention.span_token_positions(tokenizer, ex.context, spans)
    ids = tokenizer(retention.templated_context(tokenizer, ex.context), add_special_tokens=False,
                    return_tensors="pt")["input_ids"].to(model.device)
    keys = {}
    hooks = [layer.self_attn.k_proj.register_forward_hook(
        lambda _m, _i, out, li=li: keys.__setitem__(li, out[0].detach()))
        for li, layer in enumerate(model.model.layers)]
    try:
        emb = model.get_input_embeddings()(ids)[0]
        model(input_ids=ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    pool = lambda states: torch.stack([states[p].float().mean(0) for p in positions])
    return target, pool(emb), [pool(keys[li]) for li in sorted(keys)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=10, help="examples per tier and depth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if device == "cuda"
                                                 else torch.float32).to(device).eval()
    examples = tasks.synthetic_set(n_per_cell=args.n, n_records=200, seed=args.seed)
    # pass 1: the average key over every record of every example (all tiers), per layer, used for centering
    emb_sum, key_sum, count = 0, None, 0
    for ex in examples:
        _, emb, keys = pooled_keys(model, tokenizer, ex)
        emb_sum = emb_sum + emb.sum(0)
        key_sum = [k.sum(0) for k in keys] if key_sum is None else [a + k.sum(0) for a, k in zip(key_sum, keys)]
        count += emb.shape[0]
    emb_mean, key_mean = emb_sum / count, [k / count for k in key_sum]
    # pass 2: cosines, raw and centered on the global mean
    rows = []
    for ex in examples:
        target, emb, keys = pooled_keys(model, tokenizer, ex)
        rows.append({"id": ex.id, "tier": ex.meta["tier"],
                     "embedding": _cos_to_target(emb, target).mean().item(),
                     "embedding_centered": _cos_to_target(emb, target, emb_mean).mean().item(),
                     "layers": [_cos_to_target(k, target).mean().item() for k in keys],
                     "layers_centered": [_cos_to_target(k, target, m).mean().item() for k, m in zip(keys, key_mean)]})
    summary = {}
    for tier in sorted({r["tier"] for r in rows}):
        tr = [r for r in rows if r["tier"] == tier]
        avg = lambda f: [sum(r[f][i] for r in tr) / len(tr) for i in range(len(tr[0][f]))]
        layers, centered = avg("layers"), avg("layers_centered")
        summary[tier] = {"n": len(tr), "embedding": sum(r["embedding"] for r in tr) / len(tr),
                         "embedding_centered": sum(r["embedding_centered"] for r in tr) / len(tr),
                         "layer_mean": sum(layers) / len(layers), "layers": layers,
                         "layer_mean_centered": sum(centered) / len(centered), "layers_centered": centered}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "summary": summary, "rows": rows}, indent=1))
    for tier, s in summary.items():
        print(tier, f"embedding {s['embedding']:.3f} (centered {s['embedding_centered']:.3f})",
              f"keys layer mean {s['layer_mean']:.3f} (centered {s['layer_mean_centered']:.3f},"
              f" range {min(s['layers_centered']):.3f} to {max(s['layers_centered']):.3f})")


if __name__ == "__main__":
    main()
