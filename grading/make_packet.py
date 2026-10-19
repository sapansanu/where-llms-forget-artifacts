"""Build a blind grading packet for a second grader.

Part A: 15 of the 60 classified studies, drawn at random within storage substrate. For each, the reading notes
(summary, setup, measurements, findings with quoted evidence, limitations) and a link to the paper, but no grades,
no term labels and no notes on our framing. The grader assigns one grade (M, B, I, U or --) per term using only the
rules of Sections 2.1 and 2.2, which are reproduced verbatim.
Part B: 40 screening candidates, 20 the LLM kept and 20 it excluded, shuffled, labels hidden. The grader marks each
relevant or not relevant to the survey's scope.

Writes packet.tex/packet.pdf, answer_sheet.csv and _key_do_not_share.json (our grades and labels, for scoring only).

    python3 review/blind_grading/make_packet.py
"""

import csv
import json
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import argparse

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=20261011)
ap.add_argument("--exclude-key", default="", help="key file of an earlier round: its studies and papers are not redrawn")
ap.add_argument("--out", default=str(HERE))
ARGS = ap.parse_args()
OUT = Path(ARGS.out)
OUT.mkdir(parents=True, exist_ok=True)
SEED = ARGS.seed
N_STUDIES, N_SCREEN = 15, 40
TERMS = ["capacity", "addressing", "interference", "write"]
TERM_NAMES = {"capacity": "Capacity", "addressing": "Addressing", "interference": "Interference",
              "write": "Write distortion"}

GLOSSARY = [
    ("LLM", "large language model, e.g. the model behind a chatbot."),
    ("Context", "the text the model reads before answering (documents, chat history, the question)."),
    ("KV cache", "the model's working memory of the context: one stored entry (a key and a value) per token."),
    ("Token", "a word piece; the unit the model reads. A page of text is roughly 500 tokens."),
    ("Eviction", "deleting cache entries to save memory. SnapKV, H2O, TOVA and StreamingLLM are eviction policies."),
    ("Quantization", "storing numbers with fewer bits (e.g. 2 instead of 16), which saves memory but adds rounding error."),
    ("Prompt compression", "shortening the input text or replacing it with a few learned vectors before the model reads it."),
    ("Retrieval / RAG", "looking up a few passages in an external collection (index) and giving them to the model."),
    ("Top-k", "keeping only the k highest-scoring items (passages or cache entries)."),
    ("ANN search", "approximate nearest-neighbor search: a fast lookup that can miss the true best match."),
    ("Agent memory", "notes, summaries or extracted facts an AI agent keeps about past steps or conversations."),
    ("Budget", "how much memory is allowed: tokens, cache entries, bits per entry, or retrieved items k."),
    ("Full cache", "the uncompressed reference run: the model reads everything."),
    ("Baseline / ablation", "a comparison run; an ablation removes one component to see what it contributed."),
    ("Exact match, F1, accuracy", "answer-quality scores; higher is better."),
    ("Perplexity", "how surprised the model is by text; lower is better."),
    ("Needle in a haystack", "a test that hides one fact in a long text and asks for it."),
    ("Abstract only", "we could read only the paper's summary, not the full paper (marked in the notes)."),
]


def tex_escape(s: str) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


def rules_text() -> str:
    """Sections 2.1 and 2.2 of the paper, verbatim apart from citations and cross-references."""
    s = (ROOT / "tex/sections/02_associative_memory_primer_and_recap_of_a.tex").read_text()
    s = s[s.index(r"\subsection{The Associative Store"):s.index(r"\subsection{Architecture-Level")]
    s = re.sub(r"~?\\cite\{[^}]*\}", "", s)
    s = re.sub(r"Table~\\ref\{tab:master\}", "the paper's Table 1", s)
    s = re.sub(r"Section~\\ref\{[^}]*\}", "the paper", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    return s


def card_block(card: dict) -> str:
    def items(xs):
        return "\n".join(rf"\item {tex_escape(x)}" for x in xs if x)
    parts = [rf"\textbf{{Summary.}} {tex_escape(card.get('summary', ''))}",
             rf"\textbf{{What they did.}} {tex_escape(card.get('mechanism', ''))}",
             rf"\textbf{{How they measured.}} {tex_escape(card.get('measurement', ''))}"]
    finds = [f"{f.get('claim', '')} Evidence: \u201c{f.get('evidence', '')}\u201d ({f.get('where', '')})"
             for f in card.get("findings", [])]
    if finds:
        parts.append(r"\textbf{Findings.}" + "\n\\begin{itemize}\\itemsep0pt\n" + items(finds) + "\n\\end{itemize}")
    lim = card.get("limitations")
    if lim:
        lim = lim if isinstance(lim, list) else [lim]
        parts.append(r"\textbf{Limitations.}" + "\n\\begin{itemize}\\itemsep0pt\n" + items(lim) + "\n\\end{itemize}")
    return "\n\n".join(parts)


def main() -> None:
    reg = json.loads((ROOT / "framing/registry.json").read_text())
    cards = json.loads((ROOT / "notes/cards.json").read_text())
    cards = cards if isinstance(cards, dict) else {c["key"]: c for c in cards}
    corpus = {p["key"]: p for p in json.loads((ROOT / "corpus/corpus.json").read_text())}
    rng = random.Random(SEED)
    prev = json.loads(Path(ARGS.exclude_key).read_text()) if ARGS.exclude_key else {}
    skip_studies = set(prev.get("studies", {}))
    skip_titles = set(prev.get("screening_titles", {}).values())

    # Part A: allocate 15 draws across substrates in proportion to their size (largest remainder), then draw
    by_sub = defaultdict(list)
    for k, r in sorted(reg.items()):
        if k != "thiswork" and k not in skip_studies:
            by_sub[r["substrate"]].append(k)
    total = sum(len(v) for v in by_sub.values())
    quota = {s: N_STUDIES * len(v) / total for s, v in by_sub.items()}
    alloc = {s: int(q) for s, q in quota.items()}
    for s in sorted(quota, key=lambda s: quota[s] - alloc[s], reverse=True)[:N_STUDIES - sum(alloc.values())]:
        alloc[s] += 1
    studies = sorted(k for s in sorted(by_sub) for k in rng.sample(by_sub[s], alloc[s]))
    rng.shuffle(studies)

    # Part B: 20 kept and 20 excluded screening candidates, shuffled
    cands = json.loads((ROOT / "corpus/candidates.json").read_text())
    cands = [c for c in cands if c["title"] not in skip_titles]
    kept = [c for c in cands if c["role"] != "exclude"]
    excl = [c for c in cands if c["role"] == "exclude"]
    screen = rng.sample(kept, N_SCREEN // 2) + rng.sample(excl, N_SCREEN // 2)
    rng.shuffle(screen)

    key = {"studies": {k: reg[k]["grades"] for k in studies},
           "screening": {f"S{i + 1}": c["role"] != "exclude" for i, c in enumerate(screen)},
           "screening_titles": {f"S{i + 1}": c["title"] for i, c in enumerate(screen)}, "seed": SEED}
    (OUT / "_key_do_not_share.json").write_text(json.dumps(key, indent=1))

    with open(OUT / "answer_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "study or paper", "Capacity", "Addressing", "Interference", "Write distortion",
                    "relevant (yes/no, Part B only)", "note"])
        for i, k in enumerate(studies, 1):
            w.writerow([f"A{i}", reg[k]["short_name"], "", "", "", "", "", ""])
        for i, c in enumerate(screen, 1):
            w.writerow([f"S{i}", c["title"], "", "", "", "", "", ""])

    body = [r"""\documentclass[10pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,enumitem,parskip,url}
\setlist{itemsep=1pt}
\title{Blind Grading Packet}
\author{For the second grader of ``Why Compressed LLM Memory Forgets''}
\date{}
\begin{document}
\maketitle
\section*{Instructions}
Thank you for helping. This packet has two parts and takes about two to three hours in total.

\textbf{Part A (about 2 hours).} For each of the 15 studies, read our reading notes (and, if you wish, the paper at
the link) and assign one grade to each of the four terms: Capacity, Addressing, Interference and Write distortion.
Use only the rules in Section 2 of this packet. Grades:
M (measured), B (bounded), I (inferred), U (unassigned: the evidence mixes terms), or -- (the study does not address
this term). Remember two caps: Capacity can only be B, U or --; a study marked ``read in abstract only'' can get at
most I (U for Capacity).

\textbf{Part B (about 30 minutes).} For each of the 40 papers, read the title and abstract and answer: is it relevant
to a survey of information loss in compressed memory of LLM systems (KV caches, prompt compression, retrieval, agent
memory, or associative recall in sequence models)? Answer yes or no.

\textbf{Please work alone.} Do not look at the paper's Table 1, the author's grades, or any earlier spot-check sheet,
and do not discuss items with the author until you have finished. Write your answers in \texttt{answer\_sheet.csv}
(or on paper, one line per item). Use the note column for anything you were unsure about. There are no trick
questions; disagreement is useful information.

\section*{1. Plain-language glossary}
\begin{description}[itemsep=0pt,font=\normalfont\bfseries]
"""]
    body += [rf"\item[{tex_escape(a)}] {tex_escape(b)}" for a, b in GLOSSARY]
    body += [r"\end{description}", r"\section*{2. The rules (Sections 2.1 and 2.2 of the paper)}",
             r"\renewcommand{\subsection}[1]{\par\medskip\noindent\textbf{##1}\par}", rules_text(),
             r"\clearpage\section*{3. Part A: studies to grade}"]
    for i, k in enumerate(studies, 1):
        p, c, r = corpus.get(k, {}), cards.get(k, {}), reg[k]
        link = f"https://arxiv.org/abs/{p['arxiv_id']}" if p.get("arxiv_id") else (
            f"https://doi.org/{p['doi']}" if p.get("doi") else "")
        read = "full text" if c.get("read") == "full text" else "abstract only"
        body += [rf"\subsection*{{A{i}. {tex_escape(r['short_name'])}: {tex_escape(p.get('title', ''))} ({p.get('year', '')})}}",
                 rf"\textit{{Read in {read}.}} \url{{{link}}}" + "\n", card_block(c),
                 r"\medskip\noindent\textbf{Your grades:} Capacity \underline{\hspace{1.2cm}} \quad Addressing "
                 r"\underline{\hspace{1.2cm}} \quad Interference \underline{\hspace{1.2cm}} \quad Write distortion "
                 r"\underline{\hspace{1.2cm}}", r"\bigskip\hrule"]
    body += [r"\clearpage\section*{4. Part B: screening check}"]
    for i, c in enumerate(screen, 1):
        body += [rf"\subsection*{{S{i}. {tex_escape(c['title'])} ({c.get('year', '')})}}",
                 tex_escape((c.get("abstract") or "")[:1500]),
                 r"\par\noindent\textbf{Relevant?} yes / no"]
    body += [r"\end{document}"]
    (OUT / "packet.tex").write_text("\n".join(body))
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "packet.tex"], cwd=OUT, capture_output=True)
    for ext in ("aux", "log", "out"):
        (OUT / f"packet.{ext}").unlink(missing_ok=True)
    print("studies:", [reg[k]["short_name"] for k in studies])
    print("substrate allocation:", alloc)
    print("wrote", OUT / "packet.pdf", OUT / "answer_sheet.csv")


if __name__ == "__main__":
    main()
