"""Which KV entries did a press keep? Replaces ScorerPress.compress (which SnapKV, TOVA and StreamingLLM all inherit)
with an identical version that also records, per layer, a boolean mask of kept positions (kv_heads x seq). Scoring and
pruning are exactly kvpress's: top-n_kept of the press's own scores. Optionally, `protect` positions get the highest score
in every layer and head, so they are always kept while the budget (n_kept) stays the same: the intervention that tests
whether evicting an item caused an error."""

import random
import zlib

import torch
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import compute_n_kept


class RetentionRecorder:
    def __init__(self):
        self.masks: dict[int, torch.Tensor] = {}  # layer index -> bool (kv_heads, seq)
        self.protect: list[int] = []  # token positions forced into the kept set (same budget)
        self._original = None

    def __enter__(self):
        recorder, original = self, ScorerPress.compress
        self._original = original

        def compress(press, module, hidden_states, keys, values, attentions, kwargs):
            if press.compression_ratio == 0:
                return keys, values
            scores = press.score(module, hidden_states, keys, values, attentions, kwargs)
            if recorder.protect:
                idx = torch.tensor([p for p in recorder.protect if p < scores.shape[-1]], device=scores.device)
                scores = scores.clone()
                scores[..., idx] = torch.finfo(scores.dtype).max
            n_kept = compute_n_kept(keys.shape[2], press.compression_ratio)
            indices = scores.topk(n_kept, dim=-1).indices
            mask = torch.zeros(scores.shape[1], scores.shape[2], dtype=torch.bool, device=scores.device)
            mask.scatter_(1, indices[0], True)  # batch size 1
            recorder.masks[int(getattr(module, "layer_idx", len(recorder.masks)))] = mask.cpu()
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
            return keys.gather(2, indices).contiguous(), values.gather(2, indices).contiguous()

        ScorerPress.compress = compress
        return self

    def __exit__(self, *exc):
        ScorerPress.compress = self._original
        return False

    def reset(self, protect: list[int] | None = None) -> None:
        self.masks = {}
        self.protect = protect or []

    def coverage(self, token_positions: list[int]) -> dict | None:
        """Three views of whether an item survived, over layers and KV heads:
        mean: average share of the item's tokens kept; full_heads: share of (layer, head) pairs that kept every
        token; any_head_layers: share of layers in which at least one head kept every token."""
        if not self.masks or not token_positions:
            return None
        idx = torch.tensor(token_positions)
        masks = [m for m in self.masks.values() if idx.max() < m.shape[1]]
        if not masks:
            return None
        per_head = [m[:, idx] for m in masks]  # (heads, span) per layer
        return {"mean": sum(h.float().mean().item() for h in per_head) / len(per_head),
                "full_heads": sum(h.all(dim=1).float().mean().item() for h in per_head) / len(per_head),
                "any_head_layers": sum(h.all(dim=1).any().item() for h in per_head) / len(per_head)}

    def retention(self, token_positions: list[int]) -> float | None:
        """Mean over layers and KV heads of the share of `token_positions` that were kept. None = no compression."""
        if not self.masks or not token_positions:
            return None
        idx = torch.tensor(token_positions)
        shares = [m[:, idx].float().mean().item() for m in self.masks.values() if idx.max() < m.shape[1]]
        return sum(shares) / len(shares) if shares else None


def templated_context(tokenizer, context: str) -> str:
    """The exact context string kvpress tokenizes (chat template applied, question part split off)."""
    if tokenizer.chat_template is None:
        return (getattr(tokenizer, "bos_token", "") or "") + context
    separator = "#" * (len(context) + 10)
    full = tokenizer.apply_chat_template([{"role": "user", "content": context + separator}],
                                         add_generation_prompt=True, tokenize=False)
    return full.split(separator)[0]


def span_token_positions(tokenizer, context: str, spans: list[tuple[int, int]]) -> list[list[int]]:
    """Token positions (in kvpress's context_ids) covering each character span of `context`."""
    templated = templated_context(tokenizer, context)
    shift = templated.index(context)
    enc = tokenizer(templated, add_special_tokens=False, return_offsets_mapping=True)
    out = []
    for a, b in spans:
        a, b = a + shift, b + shift
        out.append([i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < b and e > a])
    return out


def sham_positions(item: list[int], allowed: list[int], seed_key: str) -> list[int]:
    """A contiguous block of len(item) token positions inside `allowed` that does not overlap the item: the control
    for protection (same number of protected tokens, no answer-bearing content). Seeded by the example id."""
    item_set, allowed = set(item), sorted(allowed)
    n = len(item)
    starts = [i for i in range(len(allowed) - n + 1)
              if allowed[i + n - 1] - allowed[i] == n - 1 and not item_set.intersection(allowed[i:i + n])]
    if not starts:
        return []
    i = random.Random(zlib.crc32(seed_key.encode())).choice(starts)
    return allowed[i:i + n]
