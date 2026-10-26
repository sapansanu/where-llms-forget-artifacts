# Why Compressed LLM Memory Forgets: code and data

Companion code and data for the paper *Why Compressed LLM Memory Forgets: An Associative-Memory Survey* by Sapan Tiwari.

The repository has two parts:

- `case_study/` holds the code, logged runs and analysis behind the paper's experiments:
  - Section 9 (Table 8 and Fig. 2): protecting the answer-bearing item, or a count-matched sham span, inside compressed
    KV caches, with Qwen2.5-7B-Instruct and, for Task 1, Mistral-7B-Instruct-v0.3;
  - Section 7.6 (Table 5): approximate (HNSW) and compressed (product-quantization) retrieval indexes against exact
    search, with the same protect-versus-sham repair.
- `literature/` holds the classification registry behind Table 1 and the cited references.

## Case study

The reader is Qwen2.5-7B-Instruct in bfloat16. SnapKV, TOVA and StreamingLLM come from
[kvpress](https://github.com/NVIDIA/kvpress) 0.5.5 and compress the context's KV cache at prefill to 25% or 10% of its
entries. Task 1 is synthetic key-value recall. Task 2 uses 98 HotpotQA questions in the distractor setting.
Supplementary Appendix B of the paper gives the full setup.

### Layout

| Path | Contents |
| --- | --- |
| `memloss/tasks.py` | Task generators: synthetic key-value records (Task 1) and HotpotQA contexts (Task 2). Both are deterministic given the seed. |
| `memloss/retention.py` | A patch to kvpress's `ScorerPress.compress`. It records per-head retention and protects a token span by giving it the maximum finite score before top-k, at the same budget. It also places the sham span. |
| `memloss/run.py` | Runs one shard of a task under each compression condition and writes one JSON line per query. |
| `memloss/scoring.py` | Scoring: code containment (Task 1); exact match after HotpotQA normalization (Task 2). |
| `memloss/analyze.py` | Paired C→W analysis, repair rates with Wilson intervals, McNemar tests, sham controls, the load sweep and clustered logistic regressions. |
| `memloss/paper_table.py` | Writes the paper's Table 8 from the logged runs. |
| `memloss/figure.py` | Draws the paper's Fig. 2 from the logged runs. |
| `memloss/second_reader.py` | The Task 1 replication with Mistral-7B: repairs, breaks and the load sweep (Section 9). |
| `memloss/keysim.py` | Key similarity inside the reader: input embeddings and pre-RoPE attention keys per layer. |
| `memloss/ann.py` | The retrieval experiment: exact, HNSW and quantized indexes over HotpotQA paragraphs, the reader, and the protect/sham repairs. |
| `memloss/ann_analyze.py` | Summarizes the retrieval experiment and writes the paper's Table 5. |
| `memloss/paper_stats.py` | Computes the remaining statistics quoted in the text (pooled repairs with intervals that resample whole questions, neighbour and closeness analyses, retention thresholds, gap recovery, protected-store bounds, Task 2 descriptives, key similarities). |
| `modal_app.py` | Runs shards on Modal A100-80GB GPUs. |
| `tests/` | Tests for generator reproducibility across processes, protection at the same budget, sham placement and answer spans. |
| `results/v2/` | Task 1 logs: main runs (`v2-A-*`), protected (`v2-protect-A-*`), sham (`v2-sham-A-*`) and the full-cache load sweep at 700 and 1,400 records (`v2-sweep-N*`). |
| `results/full/` | Task 2 logs: main runs (`B-*`), protected (`protect-B-*`), sham (`sham-B-*`) and the closed-book control (`closed-B-*`). |
| `results/mistral/` | Task 1 logs for Mistral-7B: main runs, protected, sham and the full-cache load sweep. |
| `results/keysim/` | Key-similarity results for both readers. |
| `results/ann/` | Retrieval-experiment logs: `ann-n500.jsonl` (exact, HNSW, PQ) and `ann-eq48-n500.jsonl` (48-byte codes). |
| `results/final/` | Analysis outputs used in the paper (CSV and JSON), plus the figure. |

Each log line records the example id, task, similarity tier, position, policy (`press`), kept share (`1 - ratio`), the
query-knowledge setting (`qk`: `aware` when the question is also in the context before compression), the model's
answer, correctness and token retention of the answer-bearing item. Lines from protected runs carry `protected: true`;
lines from sham runs carry `sham: true`. The main runs predate these flags, so their lines lack both fields.

### Reproducing the analysis from the logs

The analysis needs no GPU. With [uv](https://docs.astral.sh/uv/) installed:

```bash
cd case_study
uv sync
uv run pytest -q
uv run python -m memloss.analyze results/v2/v2-A-*.jsonl results/v2/v2-sweep-*.jsonl results/v2/v2-protect-A-*.jsonl results/v2/v2-sham-A-*.jsonl --out results/final/A
uv run python -m memloss.analyze results/full/B-*.jsonl results/full/closed-B-*.jsonl results/full/protect-B-*.jsonl results/full/sham-B-*.jsonl --out results/final/B
uv run python -m memloss.paper_table --out case_study_table.tex
uv run python -m memloss.figure --out results/final/case_study
uv run python -m memloss.second_reader --results results/mistral --out results/final/mistral_A_summary.json
uv run python -m memloss.ann_analyze results/ann/ann-n500.jsonl results/ann/ann-eq48-n500.jsonl --out results/final/ann --tex ann_table.tex
uv run python -m memloss.paper_stats --out results/final/paper_stats.json
```

These commands regenerate `results/final/`, the paper's Tables 4 and 7 and the statistics in its text exactly from the logs. `paper_stats` also rebuilds the Task 2 examples, so it downloads HotpotQA and the Qwen tokenizer. (On macOS, set
`KMP_DUPLICATE_LIB_OK=TRUE` for `ann_analyze`, because torch and faiss each ship an OpenMP runtime.)

### Rerunning the experiments

The runs used Modal (`uv run modal setup` once, then the commands below). The reported runs took about nine A100 GPU-hours.

```bash
# Task 1: main runs, protected item, sham span
modal run modal_app.py::main --task A --n 60 --shards 4 --tag v2-
modal run modal_app.py::main --task A --n 60 --shards 4 --protect --tag v2-protect-
modal run modal_app.py::main --task A --n 60 --shards 4 --sham --tag v2-sham-
# Task 1: full-cache load sweep
modal run modal_app.py::main --task A --n 60 --shards 2 --n-records 700 --conditions full --qk agnostic --tag v2-sweep-N700-
modal run modal_app.py::main --task A --n 60 --shards 2 --n-records 1400 --conditions full --qk agnostic --tag v2-sweep-N1400-
# Task 2: main runs, protected item, sham span, closed book
modal run modal_app.py::main --task B --n 100 --shards 4 --max-new-tokens 24
modal run modal_app.py::main --task B --n 100 --shards 4 --max-new-tokens 24 --protect --tag protect-
modal run modal_app.py::main --task B --n 100 --shards 4 --max-new-tokens 24 --sham --tag sham-
modal run modal_app.py::main --task B --n 100 --shards 1 --max-new-tokens 24 --conditions closed --tag closed-
# Task 1 with a second reader (Mistral-7B)
M=mistralai/Mistral-7B-Instruct-v0.3
modal run modal_app.py::main --task A --n 20 --shards 4 --model $M --tag mistral-
modal run modal_app.py::main --task A --n 20 --shards 4 --model $M --protect --tag mistral-protect-
modal run modal_app.py::main --task A --n 20 --shards 4 --model $M --sham --tag mistral-sham-
modal run modal_app.py::main --task A --n 60 --shards 2 --model $M --n-records 200 --conditions full --qk agnostic --tag mistral-sweep-N200-
modal run modal_app.py::main --task A --n 60 --shards 2 --model $M --n-records 700 --conditions full --qk agnostic --tag mistral-sweep-N700-
modal run modal_app.py::main --task A --n 60 --shards 2 --model $M --n-records 1400 --conditions full --qk agnostic --tag mistral-sweep-N1400-
# Key similarity inside each reader
modal run modal_app.py::keysim --model Qwen/Qwen2.5-7B-Instruct --n 10
modal run modal_app.py::keysim --model $M --n 10
# Retrieval experiment (Section 7.6)
modal run modal_app.py::ann --n 500
modal run modal_app.py::ann --n 500 --configs exact,pq-48,opq-48,lsh-384 --tag eq48-
modal volume get memloss-results / results/
```

To run without Modal, call `python -m memloss.run` directly on a GPU with the same arguments (see `--help`). Task 2
samples 100 validation questions with a fixed seed and drops the two yes/no questions. Decoding is greedy. GPU kernels
are not guaranteed to be bitwise deterministic, so a rerun can differ from the logs on a few rows.

An earlier version of the Task 1 generator depended on Python's set ordering, which varies between processes. Its runs
were discarded, are not included here, and are not used in the paper.

## Literature registry

`literature/screening.csv` lists all 1,458 search candidates with the LLM screen's role, the ranking score and whether each entered the corpus (Supplementary Appendix C).
`literature/registry.json` holds one record per classified study in Table 1. Each record gives the storage
substrate, budget currency, operation codes (W/S/A/R), write-time query knowledge, and an evidence grade for each term.
The grades are M (measured), B (bounded), I (inferred), U (unassigned) or "-" (not addressed). Each record also lists
the findings filed under each term and the basis for its grades. `read` records whether the study was read in full or
in abstract only. `literature/references.bib` is the paper's bibliography.

## Grading check

`grading/` holds the blind grading check reported in Section 11 and Supplementary Appendix F:
- `packet.pdf`: the packet the independent grader received (rules, glossary and reading notes, no grades).
- `answer_sheet_blank.csv`: the blank answer sheet.
- `answer_key.json`: the grades the packet was scored against.
- `score.py`: computes agreement and Cohen's kappa.
- `make_packet.py`: draws the sample and builds the packet.
- `changes_after_grading.json`: the six cells changed after checking the grader's disagreements.

`score.py` and `make_packet.py` expect the paper project's directory layout and are included for reference.

## License

The code is released under the MIT License (`LICENSE`). The logs, analysis outputs and registry are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Task 2 logs contain HotpotQA question identifiers and
model answers. HotpotQA is distributed under CC BY-SA 4.0
([Yang et al., 2018](https://hotpotqa.github.io/)).

## Citation

If you use this code or data, please cite the paper. A BibTeX entry will be added once it is published.
