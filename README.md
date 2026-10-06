# Why Compressed LLM Memory Forgets: code and data

Companion code and data for the paper *Why Compressed LLM Memory Forgets: An Associative-Memory Survey* by Sapan Tiwari.

The repository has two parts:

- `case_study/` holds the code, logged runs and analysis behind Section 9 of the paper (Table 8 and Fig. 1). The case
  study protects the answer-bearing item, or a count-matched sham span, inside compressed KV caches.
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
| `memloss/figure.py` | Draws the paper's Fig. 1 from the logged runs. |
| `modal_app.py` | Runs shards on Modal A100-80GB GPUs. |
| `tests/` | Tests for generator reproducibility across processes, protection at the same budget, sham placement and answer spans. |
| `results/v2/` | Task 1 logs: main runs (`v2-A-*`), protected (`v2-protect-A-*`), sham (`v2-sham-A-*`) and the full-cache load sweep at 700 and 1,400 records (`v2-sweep-N*`). |
| `results/full/` | Task 2 logs: main runs (`B-*`), protected (`protect-B-*`), sham (`sham-B-*`) and the closed-book control (`closed-B-*`). |
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
```

These commands regenerate `results/final/` and the paper's Table 8 exactly from the logs.

### Rerunning the experiments

The runs used Modal (`uv run modal setup` once, then the commands below). The reported runs took about six A100 GPU-hours.

```bash
# Task 1: main runs, protected item, sham span
modal run modal_app.py --task A --n 60 --shards 4 --tag v2-
modal run modal_app.py --task A --n 60 --shards 4 --protect --tag v2-protect-
modal run modal_app.py --task A --n 60 --shards 4 --sham --tag v2-sham-
# Task 1: full-cache load sweep
modal run modal_app.py --task A --n 60 --shards 2 --n-records 700 --conditions full --qk agnostic --tag v2-sweep-N700-
modal run modal_app.py --task A --n 60 --shards 2 --n-records 1400 --conditions full --qk agnostic --tag v2-sweep-N1400-
# Task 2: main runs, protected item, sham span, closed book
modal run modal_app.py --task B --n 100 --shards 4 --max-new-tokens 24
modal run modal_app.py --task B --n 100 --shards 4 --max-new-tokens 24 --protect --tag protect-
modal run modal_app.py --task B --n 100 --shards 4 --max-new-tokens 24 --sham --tag sham-
modal run modal_app.py --task B --n 100 --shards 1 --max-new-tokens 24 --conditions closed --tag closed-
modal volume get memloss-results / results/
```

To run without Modal, call `python -m memloss.run` directly on a GPU with the same arguments (see `--help`). Task 2
samples 100 validation questions with a fixed seed and drops the two yes/no questions. Decoding is greedy. GPU kernels
are not guaranteed to be bitwise deterministic, so a rerun can differ from the logs on a few rows.

An earlier version of the Task 1 generator depended on Python's set ordering, which varies between processes. Its runs
were discarded, are not included here, and are not used in the paper.

## Literature registry

`literature/registry.json` holds one record per classified study in Table 1. Each record gives the storage
substrate, budget currency, operation codes (W/S/A/R), write-time query knowledge, and an evidence grade for each term.
The grades are M (measured), B (bounded), I (inferred), U (unassigned) or "-" (not addressed). Each record also lists
the findings filed under each term and the basis for its grades. `read` records whether the study was read in full or
in abstract only. `literature/references.bib` is the paper's bibliography.

## License

The code is released under the MIT License (`LICENSE`). The logs, analysis outputs and registry are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Task 2 logs contain HotpotQA question identifiers and
model answers. HotpotQA is distributed under CC BY-SA 4.0
([Yang et al., 2018](https://hotpotqa.github.io/)).

## Citation

If you use this code or data, please cite the paper. A BibTeX entry will be added once it is published.
