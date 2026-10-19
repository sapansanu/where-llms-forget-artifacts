"""The statistics the paper's text quotes from the case study (Section 9 and Supplementary Appendix B), from the logs.

Tables 4 and 8 and Fig. 2 come from ann_analyze.py, paper_table.py and figure.py, the load sweeps and regressions from
analyze.py, and the Mistral replication from second_reader.py. This script computes the remaining numbers in the text:
pooled repair rates, the neighbour and closeness analyses, the retention-threshold check, gap recovery, the
protected-store bounds on Capacity, the TF-IDF and length descriptives of Task 2, and the key similarities.

    python -m memloss.paper_stats --out results/final/paper_stats.json

The Task 2 descriptives rebuild the (deterministic) examples, so they need the HotpotQA dataset and the Qwen tokenizer.
"""

import argparse
import collections
import glob
import json
import math
import re
import statistics

import numpy as np
import pandas as pd

from memloss import tasks
from memloss.analyze import add_answer_elsewhere, hypothesis_test, load, wilson
from memloss.second_reader import _neighbours, load_effect, repairs


def _owner_and_distance(e: tasks.Example) -> tuple[dict, dict]:
    """Map each code in a Task 1 context to its key, and each competitor key to its digit distance from the target."""
    t = e.meta["target_key"]
    owner, dist = {}, {}
    for line in e.context.split("\n"):
        k = line.split(" is ")[0].removeprefix("The access code for ")
        owner[line.rsplit(" ", 1)[1].rstrip(".")] = k
        if k != t:
            dist[k] = sum(a != b for a, b in zip(t.split("-")[1], k.split("-")[1]))
    return owner, dist


def closeness(sweep: pd.DataFrame) -> dict:
    """Near-duplicate tier, full cache: per competitor, how often a wrong answer returned that competitor's code, by
    the number of digits in which its key differs from the target's (pooled over loads)."""
    hi = sweep[sweep.tier == "high"]
    exs = {e.id: e for n in sorted(hi.n_records.unique()) for e in tasks.synthetic_set(60, n_records=int(n), seed=0)}
    competitors, returned = collections.Counter(), collections.Counter()
    for _, row in hi.iterrows():
        owner, dist = _owner_and_distance(exs[row.id])
        competitors.update(dist.values())
        if row.wrong and row.answer_source == "same word":
            returned[dist[owner[re.findall(r"\d{6}", row.answer)[0]]]] += 1
    rate = {d: returned[d] / competitors[d] for d in sorted(competitors)}
    return {"competitors": dict(sorted(competitors.items())), "returned": dict(sorted(returned.items())),
            "rate_one_vs_three_digits": rate[1] / rate[3] if rate.get(3) else None}


def task1(results: str) -> dict:
    g = lambda p: sorted(glob.glob(f"{results}/{p}"))
    main = load(g("v2/v2-A-*.jsonl") + g("v2/v2-protect-A-*.jsonl") + g("v2/v2-sham-A-*.jsonl"))
    main = main[main.n_records == 200]
    sweep = load(g("v2/v2-sweep-*.jsonl") + g("v2/v2-A-*.jsonl"))
    sweep = sweep[(sweep.press == "full") & (sweep.qk == "agnostic") & ~sweep.protected & ~sweep.sham]
    hi = sweep[sweep.tier == "high"].copy()
    exs = {e.id: e for n in sorted(hi.n_records.unique()) for e in tasks.synthetic_set(60, n_records=int(n), seed=0)}
    hi["n_ham1"] = hi.id.map(lambda i: _neighbours(exs[i]))
    # Retention thresholds: among TOVA's as-run answers whose item fell below each threshold, the share correct.
    tova = main[(main.press == "tova") & ~main.protected & ~main.sham]
    below = {}
    for col in ("retention_min", "full_heads_min", "any_head_layers_min"):
        for keep, sub in tova.groupby("keep"):
            b = sub[sub[col] < 0.5]
            below[f"{col} < 0.5, keep {keep:.0%}"] = {"n": len(b), "correct": int(b.correct.sum()),
                                                      "share": b.correct.mean()}
    m = load(g("mistral/mistral-sweep-*.jsonl"))
    m = m[(m.press == "full") & (m.qk == "agnostic")]
    return {"repairs": repairs(main), "load": load_effect(sweep),
            "one_digit_neighbours_mean": hi.groupby("n_records").n_ham1.mean().to_dict(),
            "closeness_qwen": closeness(sweep), "closeness_mistral": closeness(m),
            "tova_correct_below_threshold": below}


def _paired_diff_ci(d: pd.Series, clusters: pd.Series) -> tuple[float, float, float]:
    """Mean of a paired difference with a 95% Wald interval clustered on the HotpotQA question."""
    n, mean = len(d), d.mean()
    resid = (d - mean).groupby(clusters.values).sum()
    k = resid.size
    se = math.sqrt(k / (k - 1) * (resid ** 2).sum()) / n
    return mean, mean - 1.96 * se, mean + 1.96 * se


def task2(results: str) -> dict:
    g = lambda p: sorted(glob.glob(f"{results}/{p}"))
    df = load(g("full/B-*.jsonl") + g("full/protect-B-*.jsonl") + g("full/sham-B-*.jsonl"))
    df = df[df["item"] == "answer"]
    out = {"repairs": repairs(df)}
    # C->W repair per condition, and recovery of the gap to the full cache with each condition's protected accuracy
    # capped at the full cache's.
    comp = df[df.press != "full"].copy()
    comp["mode"] = np.where(comp.protected, "prot", np.where(comp.sham, "sham", "run"))
    full_acc = df[df.press == "full"].groupby("qk").correct.mean()
    acc = comp.groupby(["press", "keep", "qk", "mode"]).correct.mean()
    gap, rec = collections.Counter(), collections.Counter()
    for press, keep, qk in sorted({k[:3] for k in acc.index}):
        cls = "QG" if press == "streaming" else ("QA" if qk == "aware" else "PQ")
        run, prot, f = acc[(press, keep, qk, "run")], acc[(press, keep, qk, "prot")], full_acc[qk]
        gap[cls] += f - run
        rec[cls] += min(prot, f) - run
    out["gap_recovered"] = {c: rec[c] / gap[c] for c in gap}
    full = df[df.press == "full"].set_index(["id", "qk"]).correct
    w = comp.pivot_table(index=["id", "press", "keep", "qk"], columns="mode", values="correct", aggfunc="first").reset_index()
    w["full"] = [bool(full[(i, q)]) for i, q in zip(w.id, w.qk)]
    cw = w[w.full & ~w.run.astype(bool)]
    out["cw_repaired_by_condition"] = {f"{p} {k:.0%} {q}": {"n": len(s), "protect": s.prot.astype(bool).mean()}
                                       for (p, k, q), s in cw.groupby(["press", "keep", "qk"])}
    # Protected query-aware stores against the full cache with the question in the context (paired by example).
    fa = df[(df.press == "full") & (df.qk == "aware")].set_index("id")
    bounds = {}
    for (press, keep), s in comp[(comp["mode"] == "prot") & (comp.qk == "aware") & (comp.press != "streaming")].groupby(["press", "keep"]):
        s = s.set_index("id")
        d = s.correct.astype(int) - fa.loc[s.index, "correct"].astype(int)
        mean, lo, hi = _paired_diff_ci(d, fa.loc[s.index, "hotpot_id"])
        bounds[f"{press} {keep:.0%}"] = {"protected": s.correct.mean(), "diff_points": 100 * mean,
                                         "ci_points": [100 * lo, 100 * hi]}
    out["protected_vs_full_cache"] = bounds
    # Similarity effect among compressed runs with the item protected (as in analyze.hypothesis_test).
    dfe = add_answer_elsewhere(df)
    out["similarity_or_protected_only"] = hypothesis_test(dfe, kept=dfe[dfe.protected & (dfe.press != "full")])["all_conditions"]
    return out


def _tfidf(rows: list[dict]):
    """TF-IDF vectors with smooth IDF fitted on the unique paragraphs ("title: text") of the validation split."""
    tok = lambda s: re.findall(r"\b\w\w+\b", s.lower())
    paras = {f"{t}: {''.join(s)}" for r in rows for t, s in zip(r["context"]["title"], r["context"]["sentences"])}
    df = collections.Counter(w for p in paras for w in set(tok(p)))
    idf = {w: math.log((1 + len(paras)) / (1 + c)) + 1 for w, c in df.items()}
    unseen = math.log(1 + len(paras)) + 1

    def vec(s):
        v = {w: c * idf.get(w, unseen) for w, c in collections.Counter(tok(s)).items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / n for w, x in v.items()}
    return vec


def task2_descriptives(seed: int = 0) -> dict:
    from datasets import load_dataset
    from kvpress.utils import compute_n_kept
    from transformers import AutoTokenizer

    from memloss import retention
    rows = list(load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"))
    exs = [e for e in tasks.hotpot_set(rows, n_questions=100, seed=seed) if e.meta["item"] == "answer"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    by_id = {r["id"]: r for r in rows}
    vec = _tfidf(rows)
    cosine = lambda a, b: sum(x * b.get(w, 0.0) for w, x in a.items())
    lengths, n_distractors, fill = collections.defaultdict(list), {}, collections.defaultdict(list)
    cos = {"tfidf": {"support": [], "question": []}, "random": {"support": [], "question": []}}
    for e in exs:
        lengths[e.meta["tier"]].append(len(tok(e.context, add_special_tokens=False)["input_ids"]))
        for qk in ("agnostic", "aware"):
            ctx = e.context if qk == "agnostic" else f"{e.context}\n\nQuestion: {e.question}"
            n_tokens = len(tok(retention.templated_context(tok, ctx), add_special_tokens=False)["input_ids"])
            span = {i for p in retention.span_token_positions(tok, ctx, e.meta["protect_spans"]) for i in p}
            for ratio in (0.75, 0.9):
                fill[f"keep {1 - ratio:.0%}, {qk}"].append(len(span) / compute_n_kept(n_tokens, ratio))
        if e.meta["position"] != "start":
            continue
        # Paragraphs ("title: sentences") are separated by blank lines; the gold ones carry the supporting sentences.
        r = by_id[e.meta["hotpot_id"]]
        gold = set(r["supporting_facts"]["title"])
        paras = [p for p in e.context.split("\n\n") if p.strip()]
        distractors = [p for p in paras if not any(p.startswith(f"{t}: ") for t in gold)]
        n_distractors.setdefault(e.meta["hotpot_id"], {})[e.meta["tier"]] = len(distractors)
        support = " ".join(s for t, ss in zip(r["context"]["title"], r["context"]["sentences"])
                           for i, s in enumerate(ss) if (t, i) in set(zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"])))
        kind = "tfidf" if e.meta["tier"] == "high" else "random"
        for k, text in (("support", support), ("question", r["question"])):
            cos[kind][k] += [cosine(vec(text), vec(p)) for p in distractors]
    counts = collections.Counter(v.get("high") for v in n_distractors.values())
    return {"questions": len({e.meta["hotpot_id"] for e in exs}),
            "median_context_tokens": {t: statistics.median(v) for t, v in lengths.items()},
            "tfidf_distractors_per_question": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "random_matches_tfidf_count": all(v.get("high") == v.get("low") for v in n_distractors.values()),
            "tfidf_cosine_mean": {k: {q: float(np.mean(v)) for q, v in d.items()} for k, d in cos.items()},
            "median_protected_fill": {k: statistics.median(v) for k, v in fill.items()}}


def keysim(results: str) -> dict:
    out = {}
    for path in sorted(glob.glob(f"{results}/keysim/keysim-*.json")):
        d = json.load(open(path))
        s = d["summary"]
        centred = {t: s[t]["layers_centered"] for t in s}
        out[d["model"]] = {"embedding": {t: s[t]["embedding"] for t in s},
                           "keys_centered": {t: s[t]["layer_mean_centered"] for t in s},
                           "near_duplicates_closest_in_every_layer": all(
                               h > max(m, l) for h, m, l in zip(centred["high"], centred["medium"], centred["low"]))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-descriptives", action="store_true", help="skip the Task 2 descriptives (dataset, tokenizer)")
    args = ap.parse_args()
    out = {"task1_qwen": task1(args.results), "task2_qwen": task2(args.results), "key_similarity": keysim(args.results)}
    if not args.no_descriptives:
        out["task2_descriptives"] = task2_descriptives()
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
