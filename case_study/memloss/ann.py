"""Section 7 experiment: retrieval misses (ANN) versus key distortion (product quantization) at a fixed k.

The index holds every paragraph of the HotpotQA distractor validation set, embedded once with a dense retriever.
Each query retrieves k=5 paragraphs under:
  exact   - exhaustive inner-product search over float vectors (the reference)
  hnsw-E  - HNSW graph search over the same float vectors, efSearch=E: misses come from search only
  pq-M    - exhaustive search over product-quantization codes with M sub-quantizers: misses come from distorted keys
The reader (Qwen2.5-7B-Instruct) answers from the retrieved paragraphs; exact match uses the HotpotQA normalization.

For every C->W question (correct with the exact list, wrong with the approximate one) whose approximate list lost a
gold paragraph, two repairs are run at the same k:
  protect - put the lost gold paragraph(s) back in place of the lowest-ranked non-gold paragraph(s)
  sham    - put back the same number of non-gold paragraphs that the exact list had but the approximate list lost
            (or, if there are none, random non-gold paragraphs from the corpus)

    python -m memloss.ann --n 500 --out results/ann/run.jsonl
"""

import argparse
import json
import random
import zlib
from pathlib import Path

import numpy as np
import torch

from memloss import scoring

K = 5
HNSW_EF = (8, 16, 64)
PQ_M = (8, 16, 48)
EMBEDDER = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def corpus_and_queries(rows: list[dict], n: int, seed: int) -> tuple[list[tuple[str, str]], list[dict]]:
    """Unique (title, text) paragraphs over the whole split, and n sampled non-yes/no questions with the indexes of
    their gold (supporting) paragraphs."""
    paras, index_of = [], {}
    for r in rows:
        for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            key = (title, "".join(sents))
            if key not in index_of:
                index_of[key] = len(paras)
                paras.append(key)
    pool = [r for r in rows if r["answer"].lower() not in ("yes", "no")]
    queries = []
    for r in random.Random(seed).sample(pool, n):
        text = {t: "".join(s) for t, s in zip(r["context"]["title"], r["context"]["sentences"])}
        gold = [index_of[(t, text[t])] for t in sorted(set(r["supporting_facts"]["title"]))]
        queries.append({"id": r["id"], "question": r["question"], "answer": r["answer"], "gold": gold})
    return paras, queries


@torch.no_grad()
def embed(texts: list[str], device: str, batch: int = 256) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMBEDDER)
    model = AutoModel.from_pretrained(EMBEDDER).to(device).eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        cls = model(**enc).last_hidden_state[:, 0]  # bge uses the [CLS] vector
        out.append(torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy())
    return np.concatenate(out)


def build_indexes(vecs: np.ndarray, seed: int, configs: list[str] | None = None) -> dict:
    """exact, hnsw-E (graph search over float vectors), pq-M (M bytes per vector), and two other 48-byte codes for an
    equal-budget contrast: opq-48 (rotation learned before PQ) and lsh-384 (sign bits of 384 random projections)."""
    import faiss
    d = vecs.shape[1]
    want = configs or (["exact"] + [f"hnsw-{e}" for e in HNSW_EF] + [f"pq-{m}" for m in PQ_M])
    train = vecs[np.random.default_rng(seed).choice(len(vecs), min(len(vecs), 50_000), replace=False)]
    out = {}
    if "exact" in want:
        out["exact"] = faiss.IndexFlatIP(d)
        out["exact"].add(vecs)
    if any(w.startswith("hnsw") for w in want):
        hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = 40
        hnsw.add(vecs)
        for w in want:
            if w.startswith("hnsw"):
                out[w] = (hnsw, int(w.split("-")[1]))
    for w in want:
        if w.startswith("pq-"):
            ix = faiss.IndexPQ(d, int(w.split("-")[1]), 8, faiss.METRIC_INNER_PRODUCT)
        elif w.startswith("opq-"):
            m = int(w.split("-")[1])
            ix = faiss.index_factory(d, f"OPQ{m},PQ{m}", faiss.METRIC_INNER_PRODUCT)
        elif w.startswith("lsh-"):
            ix = faiss.IndexLSH(d, int(w.split("-")[1]), True)  # random rotation, then sign bits
        else:
            continue
        ix.train(train)
        ix.add(vecs)
        out[w] = ix
    return {w: out[w] for w in want}


def search(index, qvecs: np.ndarray, k: int = K) -> np.ndarray:
    if isinstance(index, tuple):
        index, ef = index
        index.hnsw.efSearch = ef
    return index.search(qvecs, k)[1]


def repair(approx: list[int], exact: list[int], gold: list[int], mode: str, n_corpus: int, seed_key: str) -> list[int]:
    """Same-k repair of an approximate list: put the lost gold paragraphs back (protect) or the same number of lost
    non-gold paragraphs (sham), each replacing the lowest-ranked non-gold paragraph of the approximate list."""
    lost_gold = [g for g in gold if g in exact and g not in approx]
    n = len(lost_gold)
    if mode == "protect":
        incoming = lost_gold
    else:
        lost_other = [e for e in exact if e not in approx and e not in gold]
        rng = random.Random(zlib.crc32(seed_key.encode()))
        while len(lost_other) < n:
            c = rng.randrange(n_corpus)
            if c not in gold and c not in approx and c not in lost_other:
                lost_other.append(c)
        incoming = lost_other[:n]
    out = list(approx)
    removable = [i for i in reversed(range(len(out))) if out[i] not in gold][:n]
    for i, new in zip(removable, incoming):
        out[i] = new
    return out


def prompt(question: str, paras: list[tuple[str, str]]) -> str:
    ctx = "\n\n".join(f"Title: {t}\n{x}" for t, x in paras)
    return f"{ctx}\n\nQuestion: {question}\nAnswer with a short phrase only."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reader", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--rows", type=int, default=0, help="use only the first N rows of the split (smoke tests)")
    ap.add_argument("--configs", default="", help="comma list of indexes (default: exact, hnsw-*, pq-*)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = list(load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"))
    rows = rows[:args.rows] if args.rows else rows
    paras, queries = corpus_and_queries(rows, args.n, args.seed)
    print(f"{len(paras)} paragraphs, {len(queries)} questions")
    pvecs = embed([f"{t}. {x}" for t, x in paras], device)
    qvecs = embed([QUERY_PREFIX + q["question"] for q in queries], device)
    indexes = build_indexes(pvecs, args.seed, args.configs.split(",") if args.configs else None)
    lists = {name: search(ix, qvecs) for name, ix in indexes.items()}

    tok = AutoTokenizer.from_pretrained(args.reader)
    reader = AutoModelForCausalLM.from_pretrained(args.reader, dtype=torch.bfloat16).to(device).eval()
    cache: dict[tuple, str] = {}

    @torch.no_grad()
    def answer(question: str, ids: list[int]) -> str:
        key = (question, tuple(ids))
        if key not in cache:
            msgs = [{"role": "user", "content": prompt(question, [paras[i] for i in ids])}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True).to(device)
            gen = reader.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
            cache[key] = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return cache[key]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for qi, q in enumerate(queries):
            exact = [int(i) for i in lists["exact"][qi]]
            exact_ans = answer(q["question"], exact)
            exact_ok = scoring.exact_match(exact_ans, q["answer"])
            for name in lists:
                ids = [int(i) for i in lists[name][qi]]
                ans = exact_ans if name == "exact" else answer(q["question"], ids)
                row = {"id": q["id"], "config": name, "gold": q["gold"], "retrieved": ids,
                       "gold_hits": sum(g in ids for g in q["gold"]), "overlap_exact": len(set(ids) & set(exact)),
                       "answer": ans, "correct": scoring.exact_match(ans, q["answer"]), "exact_correct": exact_ok}
                lost = [g for g in q["gold"] if g in exact and g not in ids]
                if name != "exact" and exact_ok and not row["correct"] and lost:
                    for mode in ("protect", "sham"):
                        fixed = repair(ids, exact, q["gold"], mode, len(paras), f"{q['id']}|{name}")
                        row[f"{mode}_correct"] = scoring.exact_match(answer(q["question"], fixed), q["answer"])
                f.write(json.dumps(row) + "\n")
            if (qi + 1) % 50 == 0:
                print(f"{qi + 1}/{len(queries)} questions")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
