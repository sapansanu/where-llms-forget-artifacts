"""Task 1 replication with a second reader (Mistral-7B-Instruct-v0.3): the numbers reported in Section 9.

Addressing: C->W rows (correct under the full cache, wrong as run) repaired by protecting the answer-bearing record
versus a count-matched sham, with Wilson intervals, McNemar's test and the answers each control breaks.
Interference: the full-cache load sweep (Fisher's exact test per load; logistic regression on log2 load within the
near-duplicate tier, with and without the number of one-digit neighbours of the target key).

    python -m memloss.second_reader --results results/mistral --out results/final/mistral_A_summary.json
"""

import argparse
import glob
import json
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import binomtest, fisher_exact

from memloss import tasks
from memloss.analyze import load, wilson


def repairs(df: pd.DataFrame) -> dict:
    full = df[df.press == "full"].set_index(["id", "qk"])["correct"]
    comp = df[df.press != "full"].copy()
    comp["mode"] = comp.apply(lambda r: "prot" if r.protected else ("sham" if r.sham else "run"), axis=1)
    w = comp.pivot_table(index=["id", "press", "ratio", "qk"], columns="mode", values="correct",
                         aggfunc="first").reset_index()
    w["full"] = [bool(full[(i, q)]) for i, q in zip(w.id, w.qk)]
    w["qa"] = (w.qk == "aware") & (w.press != "streaming")
    out = {}
    for name, sub in (("all", w), ("query_aware", w[w.qa])):
        cw = sub[sub.full & (sub.run == False)]  # noqa: E712
        ok = sub[sub.run == True]  # noqa: E712
        p, s = cw.prot.astype(bool), cw.sham.astype(bool)
        b, c = int((p & ~s).sum()), int((~p & s).sum())
        out[name] = {"c_to_w": len(cw), "protect": p.mean(), "protect_ci": wilson(int(p.sum()), len(cw)),
                     "sham": s.mean(), "sham_ci": wilson(int(s.sum()), len(cw)),
                     "mcnemar_p": binomtest(b, b + c, 0.5).pvalue if b + c else 1.0,
                     "correct_as_run": len(ok), "broken_by_protect": int((ok.prot == False).sum()),  # noqa: E712
                     "broken_by_sham": int((ok.sham == False).sum())}  # noqa: E712
    return out


def _neighbours(e: tasks.Example) -> int:
    t = e.meta["target_key"].split("-")[1]
    keys = [line.split(" is ")[0].removeprefix("The access code for ") for line in e.context.split("\n")]
    return sum(sum(a != b for a, b in zip(t, k.split("-")[1])) == 1 for k in keys if k != e.meta["target_key"])


def load_effect(df: pd.DataFrame) -> dict:
    full = df[(df.press == "full") & (df.qk == "agnostic")].copy()
    full["wrong"] = ~full.correct.astype(bool)
    per_load = {}
    for n, g in full.groupby("n_records"):
        hi, rest = g[g.tier == "high"], g[g.tier != "high"]
        eh, er = int(hi.wrong.sum()), int(rest.wrong.sum())
        per_load[int(n)] = {"high_accuracy": 1 - eh / len(hi), "high_errors": eh, "high_n": len(hi),
                            "other_errors": er, "other_n": len(rest),
                            "fisher_p": fisher_exact([[eh, len(hi) - eh], [er, len(rest) - er]])[1]}
    hi = full[full.tier == "high"].copy()
    exs = {e.id: e for n in sorted(hi.n_records.unique()) for e in tasks.synthetic_set(60, n_records=int(n), seed=0)}
    hi["n_ham1"] = hi.id.map(lambda i: _neighbours(exs[i]))
    hi["wrong_i"] = hi.wrong.astype(int)
    hi["log2_load"] = np.log2(hi.n_records)
    hi["pos"] = hi.position.astype(str)
    fits = {}
    for name, f in (("load", "wrong_i ~ log2_load + C(pos)"), ("load_given_neighbours", "wrong_i ~ log2_load + n_ham1 + C(pos)")):
        fit = smf.logit(f, data=hi).fit(disp=False)
        ci = np.exp(fit.conf_int().loc["log2_load"])
        fits[name] = {"or_per_doubling": float(np.exp(fit.params["log2_load"])), "ci": [float(ci[0]), float(ci[1])],
                      "p": float(fit.pvalues["log2_load"])}
    errs = hi[hi.wrong]
    same_word = int((errs.answer_source == "same word").sum()) if "answer_source" in errs else None
    return {"per_load": per_load, "trend": fits, "high_errors": len(errs), "same_word_errors": same_word}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/mistral")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    g = lambda p: sorted(glob.glob(f"{args.results}/{p}"))
    main_runs = load(g("mistral-A-*.jsonl") + g("mistral-protect-A-*.jsonl") + g("mistral-sham-A-*.jsonl"))
    sweep = load(g("mistral-sweep-*.jsonl"))
    out = {"model": "mistralai/Mistral-7B-Instruct-v0.3", "repairs": repairs(main_runs), "load": load_effect(sweep)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
