"""Case-study figure (single IEEE column), drawn from the logged runs.

(a) Task A, full cache: accuracy as the number of records grows, for near-duplicate keys vs unrelated or same-category
    keys (Wilson 95% intervals).
(b) Task B, question in the context: exact-match accuracy under each eviction condition, as run and with the
    answer-bearing sentences protected at the same budget; the dashed line is the full cache. SnapKV and TOVA are
    query-aware (QA) here; StreamingLLM ignores content and stays query-agnostic (QG).

    uv run python -m memloss.figure --out results/final/case_study
"""

import argparse
import glob

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from memloss.analyze import load, load_sweep
from memloss.paper_table import qk_class

INTACT, LOSS, NEUTRAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"snapkv": "SnapKV", "tova": "TOVA", "streaming": "StreamingLLM"}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=2.5)


def panel_sweep(ax, results: str) -> None:
    df = load(sorted(glob.glob(f"{results}/v2/v2-A-*.jsonl")) + sorted(glob.glob(f"{results}/v2/v2-sweep-*.jsonl")))
    t = load_sweep(df[df["qk"] == "agnostic"])
    high = t[t["tier"] == "high"]
    rest = t[t["tier"] != "high"].sort_values("accuracy").groupby("n_records", as_index=False).first()
    ax.errorbar(rest["n_records"], 100 * rest["accuracy"], yerr=[100 * (rest["accuracy"] - rest["acc_lo"]),
                100 * (rest["acc_hi"] - rest["accuracy"])], color=INTACT, lw=1.5, marker="o", ms=4, capsize=2,
                elinewidth=0.8, zorder=3, label="Unrelated or same-category keys")
    ax.errorbar(high["n_records"], 100 * high["accuracy"], yerr=[100 * (high["accuracy"] - high["acc_lo"]),
                100 * (high["acc_hi"] - high["accuracy"])], color=LOSS, lw=1.5, marker="s", ms=4, capsize=2,
                elinewidth=0.8, zorder=4, label="Near-duplicate keys")
    ax.set_xscale("log", base=2)
    ax.set_xticks([200, 700, 1400], ["200", "700", "1,400"])
    ax.minorticks_off()
    ax.set_xlim(170, 1650)
    ax.set_ylim(70, 102)
    ax.set_xlabel("Records in context (log scale)", color=INK)
    ax.set_ylabel("Accuracy, full cache (%)", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="lower left", frameon=False, handlelength=1.6, labelcolor=INK)
    ax.text(-0.27, 1.04, "(a)", transform=ax.transAxes, color=INK, fontsize=9, fontweight="bold")


def panel_protect(ax, results: str) -> None:
    g = lambda pat: sorted(glob.glob(f"{results}/full/{pat}"))
    df = load(g("B-*.jsonl") + g("protect-B-*.jsonl") + g("sham-B-*.jsonl"))
    df = df[(df["item"] == "answer") & (df["qk"] == "aware")]
    full = 100 * df[df["press"] == "full"]["correct"].mean()
    comp = df[df["press"] != "full"].copy()
    comp["mode"] = comp.apply(lambda r: "prot" if r.protected else ("sham" if r.sham else "run"), axis=1)
    acc = 100 * comp.groupby(["press", "ratio", "mode"])["correct"].mean()
    order = [(p, r) for p in ("snapkv", "tova", "streaming") for r in (0.75, 0.9)]
    ys = range(len(order))[::-1]
    for y, (p, r) in zip(ys, order):
        a, b, c = acc[(p, r, "run")], acc[(p, r, "prot")], acc[(p, r, "sham")]
        ax.plot([min(a, c), b], [y, y], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
        ax.plot(c, y, "o", mfc="white", mec=NEUTRAL, mew=1.2, ms=5, zorder=2)
        ax.plot(a, y, "o", color=LOSS, ms=5, zorder=3)
        ax.plot(b, y, "o", color=INTACT, ms=5, zorder=3)
    ax.axvline(full, color=NEUTRAL, lw=0.9, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(list(ys), [f"{NAMES[p]} {1 - r:.0%} ({qk_class(p, 'aware')})" for p, r in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(10, 65)
    ax.set_xticks([10, 20, 30, 40, 50, 60])
    ax.set_xlabel("Exact match, question in context (%)", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.5)
    handles = [Line2D([], [], ls="", marker="o", color=LOSS, ms=5, label="As run"),
               Line2D([], [], ls="", marker="o", mfc="white", mec=NEUTRAL, mew=1.2, ms=5, label="Sham protected"),
               Line2D([], [], ls="", marker="o", color=INTACT, ms=5, label="Supporting sentences protected"),
               Line2D([], [], color=NEUTRAL, lw=0.9, ls=(0, (3, 2)), label="Full cache")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.38, -0.22), ncol=2, frameon=False,
              handletextpad=0.3, columnspacing=0.9, labelcolor=INK)
    ax.text(-0.52, 1.04, "(b)", transform=ax.transAxes, color=INK, fontsize=9, fontweight="bold")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42})
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 3.8), gridspec_kw={"height_ratios": [1, 1.25], "hspace": 0.55})
    panel_sweep(a, args.results)
    panel_protect(b, args.results)
    for ax in (a, b):
        _style(ax)
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{args.out}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    print(f"wrote {args.out}.pdf and .png")


if __name__ == "__main__":
    main()
