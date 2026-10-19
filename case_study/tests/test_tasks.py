import random

from memloss import scoring, tasks


def test_synthetic_examples_mark_the_answer_value_and_respect_tiers():
    exs = tasks.synthetic_set(n_per_cell=2, n_records=200, seed=1)
    assert len(exs) == 3 * 3 * 2
    for ex in exs:
        (a, b), = ex.spans
        assert ex.context[a:b] == ex.answers[0] and len(ex.answers[0]) == 6
        keys = [line.split(" is ")[0].replace("The access code for ", "") for line in ex.context.split("\n")]
        assert len(keys) == 200 and len(set(keys)) == 200 and ex.meta["target_key"] in keys
        word = ex.meta["target_key"].split("-")[0]
        same_word = sum(k.split("-")[0] == word for k in keys) - 1
        if ex.meta["tier"] == "high":
            assert same_word == 199
        else:
            assert same_word == 0
    first = [k for k in exs if k.meta["position"] == 0.1][0]
    idx = first.context[: first.spans[0][0]].count("\n")
    assert idx == round(0.1 * 199)


def test_hotpot_builder_keeps_supporting_sentences_and_positions():
    rows = []
    for q in range(6):
        titles = [f"T{q}-{i}" for i in range(10)]
        rows.append({"id": str(q), "question": f"Q{q}?", "answer": "A",
                     "supporting_facts": {"title": [titles[0], titles[1]], "sent_id": [0, 1]},
                     "context": {"title": titles, "sentences": [[f"S{q}{i}a. ", f"S{q}{i}b. "] for i in range(10)]}})
    exs = tasks.hotpot_set(rows, n_questions=2, seed=0)
    assert len(exs) == 2 * 2 * 3
    for ex in exs:
        texts = [ex.context[a:b] for a, b in ex.spans]
        q = ex.id.split("-")[-1]
        assert len(texts) == 2 and all(t.startswith("S") for t in texts)
    start = [e for e in exs if e.meta["position"] == "start"][0]
    assert start.context.startswith(start.context.split(":")[0]) and start.spans[0][0] < 20


def test_scoring():
    assert scoring.code_correct("The code is 482913.", "482913") and not scoring.code_correct("48291", "482913")
    assert scoring.exact_match("The Eiffel Tower", "eiffel tower") and scoring.f1("Eiffel", "Eiffel Tower") > 0.6


def test_coverage_views():
    import torch
    from memloss.retention import RetentionRecorder
    rec = RetentionRecorder()
    # 2 layers x 2 heads x 6 positions; item = positions 2,3
    rec.masks = {0: torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 1, 0, 0]], dtype=torch.bool),
                 1: torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)}
    cov = rec.coverage([2, 3])
    assert cov["mean"] == (0.75 + 0.0) / 2 and cov["full_heads"] == (0.5 + 0.0) / 2 and cov["any_head_layers"] == 0.5


def test_hotpot_answer_spans_inside_support():
    row = {"id": "q1", "question": "Who composed it?", "answer": "Richard Strauss",
           "supporting_facts": {"title": ["Opera", "Composer"], "sent_id": [0, 0]},
           "context": {"title": ["Opera", "Composer", "Other"],
                       "sentences": [["It was set to music by Richard Strauss. ", "Later."],
                                     ["Richard Strauss was German."], ["Unrelated text about Strauss."]]}}
    pool_row = {"id": "q2", "question": "", "answer": "", "supporting_facts": {"title": [], "sent_id": []},
                "context": {"title": [f"P{i}" for i in range(5)], "sentences": [[f"Para {i}."] for i in range(5)]}}
    examples = tasks.hotpot_set([row, pool_row], n_questions=1, seed=1)  # seed 1 samples q1
    assert len(examples) == 6 and all(ex.meta["hotpot_id"] == "q1" for ex in examples)
    for ex in examples:
        assert ex.meta["item"] == "answer"
        assert [ex.context[a:b] for a, b in ex.spans] == ["Richard Strauss", "Richard Strauss"]
        assert all(any(sa <= a and b <= sb for sa, sb in ex.meta["support_spans"]) for a, b in ex.spans)


def test_synthetic_examples_rebuild_identically_in_a_new_process():
    import subprocess
    import sys
    code = ("from memloss import tasks; e = tasks.synthetic_set(1, n_records=50, seed=3)[4]; "
            "print(e.context + e.answers[0])")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"PYTHONHASHSEED": str(h), "PATH": ""}).stdout for h in (1, 2)}
    assert len(outs) == 1 and outs != {""}


def test_code_source_labels():
    ex = tasks.synthetic_set(1, n_records=50, seed=0, positions=(0.5,))[2]  # high tier: near-duplicate keys
    lines = ex.context.split("\n")
    other = next(line for line in lines if ex.meta["target_key"] + " " not in line)
    assert tasks.code_source(ex, ex.answers[0]) == "target"
    assert tasks.code_source(ex, other.rsplit(" ", 1)[1].rstrip(".")) == "same word"
    assert tasks.code_source(ex, "000000") == "not in context" and tasks.code_source(ex, "none") == "no code"


def test_protect_keeps_item_at_same_budget():
    import torch
    from kvpress.presses.scorer_press import ScorerPress
    from kvpress.utils import compute_n_kept
    from memloss.retention import RetentionRecorder

    class LowScoresFor(ScorerPress):  # scores the protected positions lowest, so only protection can keep them
        def score(self, module, hidden_states, keys, values, attentions, kwargs):
            s = torch.arange(keys.shape[2], dtype=torch.float32).repeat(1, keys.shape[1], 1)
            s[..., 5:8] = -1.0
            return s

    module = type("M", (), {"head_dim": 4, "layer_idx": 0})()
    keys = values = torch.randn(1, 2, 20, 4)
    press = LowScoresFor(compression_ratio=0.5)
    with RetentionRecorder() as rec:
        rec.reset()
        k, _ = press.compress(module, None, keys, values, None, {})
        assert not rec.masks[0][:, 5:8].any()
        rec.reset(protect=[5, 6, 7])
        k2, _ = press.compress(module, None, keys, values, None, {})
        assert rec.masks[0][:, 5:8].all()
    assert k.shape == k2.shape and k2.shape[2] == compute_n_kept(20, 0.5)


def test_sham_positions_disjoint_contiguous_and_reproducible():
    from memloss.retention import sham_positions
    item, allowed = [10, 11, 12, 40, 41], list(range(5, 100))
    a = sham_positions(item, allowed, "ex-1")
    assert len(a) == 5 and a == list(range(a[0], a[0] + 5)) and not set(a) & set(item)
    assert a == sham_positions(item, allowed, "ex-1") and set(a) <= set(allowed)


def test_ann_repair_is_count_matched_and_targets_lost_gold():
    from memloss.ann import repair
    exact, approx, gold = [1, 2, 3, 4, 5], [1, 9, 3, 8, 7], [2, 3]  # gold 2 lost, gold 3 kept
    protect = repair(approx, exact, gold, "protect", 100, "q")
    sham = repair(approx, exact, gold, "sham", 100, "q")
    assert len(protect) == len(sham) == len(approx)
    assert 2 in protect and 3 in protect and protect[:-1] == approx[:-1]  # lowest-ranked non-gold replaced
    assert 2 not in sham and 3 in sham and sham[-1] in (4, 5)  # a non-gold paragraph the approximate list lost


def test_ann_corpus_dedups_and_maps_gold():
    from memloss.ann import corpus_and_queries
    row = lambda i, ans: {"id": str(i), "question": f"q{i}", "answer": ans,
                          "context": {"title": ["A", "B"], "sentences": [["a1 ", "a2"], ["b1"]]},
                          "supporting_facts": {"title": ["B", "A", "B"], "sent_id": [0, 0, 0]}}
    paras, queries = corpus_and_queries([row(0, "x"), row(1, "yes"), row(2, "z")], n=2, seed=0)
    assert paras == [("A", "a1 a2"), ("B", "b1")]
    assert all(q["answer"] != "yes" and q["gold"] == [0, 1] for q in queries)
