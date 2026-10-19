"""Score the second grader's answer sheet against our grades and screening labels.

Reports raw agreement and Cohen's kappa over all graded cells (grades M/B/I/U/--), agreement per term, agreement on
whether a term is addressed at all (-- vs any grade), and agreement and kappa on the screening decisions.

    python3 review/blind_grading/score.py review/blind_grading/answer_sheet.csv
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TERMS = [("Capacity", "capacity"), ("Addressing", "addressing"), ("Interference", "interference"),
         ("Write distortion", "write")]


def norm(g: str) -> str:
    g = (g or "").strip().upper().replace("–", "-").replace("—", "-")
    return "-" if g in ("-", "--", "N/A", "NA", "NONE") else g


def kappa(pairs: list[tuple[str, str]]) -> float:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def bootstrap_ci(groups: list[list[tuple[str, str]]], n: int = 10000, seed: int = 20261019) -> tuple[float, float]:
    """95% interval for kappa, resampling whole groups (all cells of a study, or one screening decision)."""
    import random
    rng = random.Random(seed)
    ks = sorted(kappa([c for _ in groups for c in rng.choice(groups)]) for _ in range(n))
    return ks[int(0.025 * n) - 1], ks[int(0.975 * n) - 1]


def main(path: str, key_path: str = "") -> None:
    key = json.loads(Path(key_path or HERE / "_key_do_not_share.json").read_text())
    rows = {r["item"]: r for r in csv.DictReader(open(path))}
    names = {}
    reg = json.loads((HERE.parents[1] / "framing/registry.json").read_text())
    for k in key["studies"]:
        names[reg[k]["short_name"]] = k
    pairs, per_term, missing, by_study = [], {t: [] for _, t in TERMS}, [], []
    for item, r in rows.items():
        if not item.startswith("A"):
            continue
        k = names[r["study or paper"]]
        by_study.append([])
        for col, t in TERMS:
            theirs = norm(r[col])
            if not theirs:
                missing.append((item, col))
                continue
            ours = norm(key["studies"][k][t])
            pairs.append((ours, theirs))
            by_study[-1].append((ours, theirs))
            per_term[t].append(ours == theirs)
    print(f"Part A: {len(pairs)} cells, agreement {sum(a == b for a, b in pairs) / len(pairs):.0%}, "
          f"Cohen's kappa {kappa(pairs):.2f} (95% CI {'-'.join(f'{x:.2f}' for x in bootstrap_ci(by_study))}, "
          "bootstrap over studies)")
    addressed = [(a != "-", b != "-") for a, b in pairs]
    print(f"  addressed vs not: agreement {sum(a == b for a, b in addressed) / len(addressed):.0%}")
    graded = [(a, b) for a, b in pairs if a != "-" and b != "-"]
    if graded:
        print(f"  cells both graded: {len(graded)}, agreement {sum(a == b for a, b in graded) / len(graded):.0%}")
    for col, t in TERMS:
        print(f"  {col}: {sum(per_term[t])}/{len(per_term[t])}")
    print("  confusion (ours -> theirs):", dict(Counter(f"{a}->{b}" for a, b in pairs if a != b)))
    if missing:
        print("  blank cells:", missing)
    sp = [(key["screening"][i], rows[i]["relevant (yes/no, Part B only)"].strip().lower().startswith("y"))
          for i in key["screening"] if rows.get(i, {}).get("relevant (yes/no, Part B only)", "").strip()]
    if sp:
        print(f"Part B: {len(sp)} decisions, agreement {sum(a == b for a, b in sp) / len(sp):.0%}, "
              f"Cohen's kappa {kappa([(str(a), str(b)) for a, b in sp]):.2f} "
              f"(95% CI {'-'.join(f'{x:.2f}' for x in bootstrap_ci([[(str(a), str(b))] for a, b in sp]))})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "answer_sheet.csv"), sys.argv[2] if len(sys.argv) > 2 else "")
