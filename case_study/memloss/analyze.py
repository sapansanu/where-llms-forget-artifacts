"""Turn logged runs into the case-study numbers.

For every condition and similarity tier:
- accuracy (with a Wilson 95% interval),
- T: the share of queries whose answer-bearing item survived eviction (retention >= threshold),
- errors with the item evicted (Addressing: the read cannot reach it),
- errors with the item kept (the item is present: positional under-use or Interference).
The similarity-attributable part of the kept-but-wrong errors, P(wrong | T, tier) - P(wrong | T, low), is the
Interference estimate the hypothesis concerns. A logistic regression on kept items tests whether error rises with
competitor similarity at fixed load, controlling for position and condition.

    uv run python -m memloss.analyze results/A-*.jsonl --out results/summary_A
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

TIER_ORDER = {"low": 0, "medium": 1, "high": 2}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def load(paths: list[str]) -> pd.DataFrame:
    rows = [json.loads(line) for p in paths for line in open(p)]
    df = pd.DataFrame(rows)
    df["protected"] = df["protected"].eq(True) if "protected" in df else False
    df["sham"] = df["sham"].eq(True) if "sham" in df else False
    df = df.drop_duplicates(subset=["id", "press", "ratio", "qk", "protected", "sham"])
    df["keep"] = 1 - df["ratio"]
    df["condition"] = df.apply(lambda r: "full cache" if r.press == "full" else f"{r.press} {r.keep:.0%}", axis=1)
    df["wrong"] = ~df["correct"].astype(bool)
    if "key_retention_min" in df:  # Task A: a record survives only if its key (address) and its value both do
        df["value_retention_min"] = df["retention_min"]
        df["retention_min"] = df[["retention_min", "key_retention_min"]].min(axis=1)
        df["full_heads_min"] = df[["full_heads_min", "key_full_heads"]].min(axis=1)
    return df


def error_sources(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Task A: where wrong answers came from, by condition, tier, and whether the record survived. A wrong code that
    belongs to a near-duplicate key ('same word') is the direct signature of interference."""
    wrong = df[df["wrong"]].assign(record_kept=lambda d: d["retention_min"].fillna(1.0) >= threshold)
    return pd.crosstab([wrong["condition"], wrong["qk"], wrong["tier"], wrong["protected"], wrong["record_kept"]],
                       wrong["answer_source"]).reset_index()


def load_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Full-cache accuracy by load (records in context) and similarity tier, with Wilson intervals."""
    full = df[df["press"] == "full"]
    out = []
    for (n_rec, tier), g in full.groupby(["n_records", "tier"]):
        k, n = int(g["correct"].sum()), len(g)
        lo, hi = wilson(k, n)
        src = g.loc[g["wrong"], "answer_source"].value_counts().to_dict() if "answer_source" in g else {}
        out.append({"n_records": n_rec, "tier": tier, "n": n, "accuracy": k / n, "acc_lo": lo, "acc_hi": hi,
                    "errors_same_word": src.get("same word", 0), "errors_other_word": src.get("other word", 0),
                    "errors_not_in_context": src.get("not in context", 0), "errors_no_code": src.get("no code", 0)})
    return pd.DataFrame(out).sort_values(["n_records", "tier"], key=lambda c: c.map(TIER_ORDER).fillna(c))


def add_answer_elsewhere(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Task B runs logged before `answer_elsewhere` existed: rebuild the (deterministic) examples and attach it."""
    if "hotpot_id" not in df or ("answer_elsewhere" in df and df["answer_elsewhere"].notna().all()):
        return df
    from datasets import load_dataset

    from memloss import tasks
    rows = list(load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"))
    n = int(df["id"].str.rsplit("-", n=1).str[1].astype(int).max()) + 1
    exs = {e.id: e for e in tasks.hotpot_set(rows, n_questions=n, seed=seed)}
    assert all(exs[i].meta["hotpot_id"] == h for i, h in zip(df["id"], df["hotpot_id"])), "examples do not match"
    return df.assign(answer_elsewhere=df["id"].map(lambda i: exs[i].meta["answer_elsewhere"]))


def load_trend(df: pd.DataFrame) -> dict:
    """Full cache: does near-duplicate (high-tier) error grow with load? Logit of wrong on log2(load) within the high
    tier, controlling for position; and, at each load, Fisher's exact test of high vs low+medium tiers."""
    from scipy.stats import fisher_exact
    full = df[df["press"] == "full"].copy()
    full["wrong_i"] = full["wrong"].astype(int)
    full["log2_load"] = np.log2(full["n_records"])
    full["pos"] = full["position"].astype(str)
    high = full[full["tier"] == "high"]
    fit = smf.logit("wrong_i ~ log2_load + C(pos)", data=high).fit(disp=False)
    lo, hi = fit.conf_int().loc["log2_load"]
    out = {"high_tier_or_per_doubling": math.exp(fit.params["log2_load"]), "or_ci": [math.exp(lo), math.exp(hi)],
           "p_value": float(fit.pvalues["log2_load"]), "n": int(len(high)), "per_load": {}}
    for n_rec, g in full.groupby("n_records"):
        h, rest = g[g["tier"] == "high"], g[g["tier"] != "high"]
        table = [[int(h["wrong"].sum()), int((~h["wrong"]).sum())], [int(rest["wrong"].sum()), int((~rest["wrong"]).sum())]]
        out["per_load"][int(n_rec)] = {"high_errors": table[0][0], "high_n": len(h), "other_errors": table[1][0],
                                       "other_n": len(rest), "fisher_p": float(fisher_exact(table)[1])}
    return out


def controls(df: pd.DataFrame, sham: pd.DataFrame) -> pd.DataFrame:
    """Protection against its count-matched sham, per condition and query knowledge, on paired rows: accuracy as run,
    with a random equal-length non-answer span protected, and with the answer-bearing item protected; plus the share of
    C->W rows (correct under the full cache, wrong as run) that each repairs."""
    keys = ["id", "press", "ratio", "qk"]
    full = df[df["press"] == "full"][["id", "qk", "correct"]].rename(columns={"correct": "full_ok"})
    run = df[(df["press"] != "full") & ~df["protected"]][keys + ["condition", "correct"]].merge(full, on=["id", "qk"])
    m = (run.merge(df[df["protected"]][keys + ["correct"]].rename(columns={"correct": "prot"}), on=keys)
            .merge(sham[keys + ["correct"]].rename(columns={"correct": "sham"}), on=keys))
    out = []
    for (cond, qk), g in m.groupby(["condition", "qk"]):
        cw = g[g["full_ok"].astype(bool) & ~g["correct"].astype(bool)]
        out.append({"condition": cond, "qk": qk, "n": len(g), "acc_run": g["correct"].mean(), "acc_sham": g["sham"].mean(),
                    "acc_protected": g["prot"].mean(), "cw_rows": len(cw),
                    "cw_repaired_protected": cw["prot"].astype(bool).mean() if len(cw) else math.nan,
                    "cw_repaired_sham": cw["sham"].astype(bool).mean() if len(cw) else math.nan})
    return pd.DataFrame(out)


def paired_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Pair each compressed run with its protected twin (same example, method, budget and query knowledge; the
    answer-bearing item forced into the cache at the same budget).
    addressing: wrong normally but right when the item is protected, so evicting the item caused the error.
    residual: wrong even with the item protected (interference from competitors, position, or other causes).
    harmed: right normally but wrong when protected (protection displaced something the answer needed)."""
    keys = ["id", "press", "ratio", "qk"]
    normal = df[(df["press"] != "full") & ~df["protected"]]
    paired = normal.merge(df[df["protected"]][keys + ["wrong"]], on=keys, suffixes=("", "_prot"))
    out = []
    for (cond, qk, tier), g in paired.groupby(["condition", "qk", "tier"]):
        w, wp = g["wrong"], g["wrong_prot"]
        out.append({"condition": cond, "qk": qk, "tier": tier, "n": len(g), "accuracy": 1 - w.mean(),
                    "accuracy_protected": 1 - wp.mean(), "addressing": (w & ~wp).mean(),
                    "residual": (w & wp).mean(), "harmed": (~w & wp).mean()})
    res = pd.DataFrame(out)
    res["tier_rank"] = res["tier"].map(TIER_ORDER)
    return res.sort_values(["condition", "qk", "tier_rank"]).drop(columns="tier_rank")


def decompose(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Threshold-based decomposition (secondary: a retention threshold does not reliably mark an item as readable)."""
    df = df[~df["protected"]].copy()
    df["T"] = df["retention_min"].fillna(1.0) >= threshold
    out = []
    for (cond, qk, tier), g in df.groupby(["condition", "qk", "tier"]):
        n, k = len(g), int(g["correct"].sum())
        lo, hi = wilson(k, n)
        kept = g[g["T"]]
        out.append({
            "condition": cond, "qk": qk, "tier": tier, "n": n, "accuracy": k / n, "acc_lo": lo, "acc_hi": hi,
            "retention_mean": g["retention_mean"].mean(), "T_rate": g["T"].mean(),
            "err_item_evicted": ((~g["T"]) & g["wrong"]).mean(),  # Addressing
            "err_item_kept": (g["T"] & g["wrong"]).mean(),
            "p_wrong_given_T": kept["wrong"].mean() if len(kept) else math.nan,
            "n_T": len(kept),
        })
    res = pd.DataFrame(out)
    low = res[res.tier == "low"].set_index(["condition", "qk"])["p_wrong_given_T"]
    res["interference_excess"] = res.apply(
        lambda r: r.p_wrong_given_T - low.get((r.condition, r.qk), math.nan), axis=1)
    res["tier_rank"] = res["tier"].map(TIER_ORDER)
    return res.sort_values(["condition", "qk", "tier_rank"]).drop(columns="tier_rank")


def hypothesis_test(df: pd.DataFrame, threshold: float = 0.5, kept: pd.DataFrame | None = None) -> dict:
    """Among queries whose answer item was present, does error rise with competitor similarity? Present = `kept` if
    given (full cache plus protected runs), else retention >= threshold."""
    if kept is None:
        kept = df[~df["protected"] & (df["retention_min"].fillna(1.0) >= threshold)]
    kept = kept.copy()
    # Similarity as the rank among the tiers present: Task B has two tiers (low, high), so one step = low -> high.
    present = sorted(df["tier"].unique(), key=TIER_ORDER.get)
    kept["sim"] = kept["tier"].map({t: i for i, t in enumerate(present)})
    kept["wrong_i"] = kept["wrong"].astype(int)
    kept["pos"] = kept["position"].astype(str)
    # Cluster unit: the HotpotQA question (shared by both tiers) or the synthetic example.
    kept["cluster"] = kept["hotpot_id"] if "hotpot_id" in kept else kept["id"]
    out = {"n_kept": int(len(kept))}
    # Task B: another copy of the answer outside the supporting sentences makes a correct answer easier; control for it.
    extra = " + C(answer_elsewhere)" if "answer_elsewhere" in kept else ""
    for name, data, formula in [
        ("full_cache", kept[kept.press == "full"], "wrong_i ~ sim + C(pos) + C(qk)" + extra),
        ("all_conditions", kept, "wrong_i ~ sim + C(pos) + C(condition) + C(qk)" + extra),
    ]:
        if data["wrong_i"].nunique() < 2:
            out[name] = {"note": "no variation in errors; model not estimable", "n": int(len(data))}
            continue
        try:
            # The same question appears under many conditions and positions: cluster standard errors on it.
            fit = smf.logit(formula, data=data).fit(disp=False, warn_convergence=False, cov_type="cluster",
                                                    cov_kwds={"groups": pd.factorize(data["cluster"])[0]})
        except Exception as e:  # e.g. perfect separation in a small cell
            out[name] = {"note": f"model not estimable: {type(e).__name__}", "n": int(len(data))}
            continue
        lo, hi = fit.conf_int().loc["sim"]
        if not fit.mle_retvals.get("converged", False) or max(abs(lo), abs(hi)) > 30:
            out[name] = {"note": "model did not converge (separation); not estimable", "n": int(len(data))}
            continue
        out[name] = {"n": int(len(data)), "clusters": int(data["cluster"].nunique()), "odds_ratio_per_tier": math.exp(fit.params["sim"]),
                     "or_ci": [math.exp(lo), math.exp(hi)], "p_value": float(fit.pvalues["sim"])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = load(args.paths)
    excluded = 0
    if "item" in df:  # Task B: keep the primary analysis on one item definition (the answer string)
        excluded = int((df["item"] == "sentence").sum())
        df = df[df["item"] != "sentence"]
        df = add_answer_elsewhere(df)
    closed = df[df["press"] == "closed"]  # closed-book control: the question with no context at all
    df = df[df["press"] != "closed"]
    sham = df[df["sham"]]  # count-matched control for protection: analyzed only in paired form (controls table)
    df = df[~df["sham"]]
    sweep = df
    if "n_records" in df and df["n_records"].nunique() > 1:  # the load sweep is tested separately (load_trend)
        df = df[df["n_records"] == df["n_records"].mode()[0]]
    table = decompose(df, args.threshold)
    sensitivity = {}
    if df["protected"].any():  # primary: the item is present by construction (full cache, or protected at budget)
        test = hypothesis_test(df, kept=df[(df["press"] == "full") | df["protected"]])
        test["design"] = "item present by construction: full cache or protected at the same budget"
        paired_decomposition(df).to_csv(f"{out}_paired.csv", index=False)
        sensitivity[f"threshold mean>={args.threshold}"] = hypothesis_test(df, args.threshold)
    else:
        test = hypothesis_test(df, args.threshold)
    sensitivity.update({f"mean>={t}": hypothesis_test(df, t) for t in (0.25, 0.75)})
    if "answer_elsewhere" in df:  # only examples whose answer occurs nowhere but in the supporting sentences
        sub = df[~df["answer_elsewhere"].astype(bool)].drop(columns="answer_elsewhere")
        sensitivity["no_other_copy"] = hypothesis_test(
            sub, args.threshold, kept=sub[(sub["press"] == "full") | sub["protected"]] if sub["protected"].any() else None)
    for col in ("full_heads_min", "any_head_layers_min"):  # alternative definitions of "the item survived"
        if col in df:
            alt = df.assign(retention_min=df[col])
            sensitivity[f"{col}>=0.5"] = hypothesis_test(alt, 0.5)
    table.to_csv(f"{out}_table.csv", index=False)
    if "answer_source" in df:
        error_sources(df, args.threshold).to_csv(f"{out}_error_sources.csv", index=False)
    if sweep is not df:
        load_sweep(sweep[sweep["qk"] == "agnostic"]).to_csv(f"{out}_load_sweep.csv", index=False)
        test["load_trend"] = load_trend(sweep[sweep["qk"] == "agnostic"])
    if len(sham):
        controls(df, sham).to_csv(f"{out}_controls.csv", index=False)
    if len(closed):
        k, n = int(closed["correct"].sum()), len(closed)
        test["closed_book_accuracy"] = {"accuracy": k / n, "ci": list(wilson(k, n)), "n": n}
    json.dump({"rows": len(df), "rows_excluded_no_verbatim_answer": excluded, "models": sorted(df["model"].unique()), "threshold": args.threshold,
               "hypothesis_test": test, "sensitivity": sensitivity}, open(f"{out}_stats.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print(json.dumps(test, indent=1))


if __name__ == "__main__":
    main()
