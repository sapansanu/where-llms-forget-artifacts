"""Task generators for the case study. Every example records the character spans of its answer-bearing items, so
retention of exactly those items can be measured after KV eviction.

Task A (synthetic key-value recall): N records "The access code for <key> is <value>." One record is asked about.
Competitor keys are low / medium / high similarity to the target key; record format is fixed, load (N) is fixed within
a run (and swept across runs), and the target's position is balanced.

Task B (HotpotQA distractor setting): the two gold paragraphs plus eight distractors that are either the dataset's
TF-IDF-retrieved paragraphs (similar to the question) or random paragraphs from other questions. The answer-bearing
item is the answer string where it occurs inside the supporting-fact sentences (or the sentences themselves when it does
not occur verbatim); retention of the whole supporting sentences is logged as a secondary measure."""

import random
import re
from dataclasses import dataclass, field

BIRDS = ["Falcon", "Heron", "Osprey", "Kestrel", "Harrier", "Plover", "Egret", "Ibis", "Petrel", "Shrike", "Merlin",
         "Avocet", "Bittern", "Curlew", "Dunlin", "Gannet", "Grebe", "Lapwing", "Linnet", "Martin", "Oriole", "Puffin",
         "Raven", "Siskin", "Swift", "Tern", "Thrush", "Warbler", "Wren", "Starling"]
OBJECTS = ["Teapot", "Glacier", "Ledger", "Canyon", "Lantern", "Harbor", "Quarry", "Anvil", "Bucket", "Cactus",
           "Domino", "Fjord", "Gadget", "Hammock", "Igloo", "Jigsaw", "Kettle", "Lagoon", "Magnet", "Nozzle", "Orchard",
           "Pylon", "Quiver", "Rafter", "Saddle", "Thimble", "Umbrella", "Valve", "Wicket", "Zipper"]
TIERS = ("low", "medium", "high")


@dataclass
class Example:
    id: str
    task: str
    context: str
    question: str
    answers: list[str]
    spans: list[tuple[int, int]]  # character spans of answer-bearing items in `context`
    meta: dict = field(default_factory=dict)


def _record(key: str, value: str) -> str:
    return f"The access code for {key} is {value}."


def synthetic_example(rng: random.Random, tier: str, position: float, n_records: int, idx: int) -> Example:
    # Four-digit suffixes in every tier: same key format everywhere, and enough distinct near-duplicates (8999) for
    # the load sweep.
    assert n_records - 1 <= 8999, "the high tier has 8999 distinct near-duplicate keys"
    target_word = rng.choice(BIRDS)
    target_key = f"{target_word}-{rng.randint(1000, 9999)}"
    used = {target_key}
    keys = []
    while len(keys) < n_records - 1:
        if tier == "high":  # same word, different number: near-duplicate keys
            k = f"{target_word}-{rng.randint(1000, 9999)}"
        elif tier == "medium":  # same category (birds), different word
            k = f"{rng.choice([b for b in BIRDS if b != target_word])}-{rng.randint(1000, 9999)}"
        else:  # unrelated objects
            k = f"{rng.choice(OBJECTS)}-{rng.randint(1000, 9999)}"
        if k not in used:
            used.add(k)
            keys.append(k)
    values, seen = [], set()  # a list, not a set: set order varies between processes, so examples would not rebuild
    while len(values) < n_records:
        v = str(rng.randint(100000, 999999))
        if v not in seen:
            seen.add(v)
            values.append(v)
    target_index = min(n_records - 1, int(round(position * (n_records - 1))))
    keys.insert(target_index, target_key)
    lines, spans, offset = [], [], 0
    for i, (k, v) in enumerate(zip(keys, values)):
        line = _record(k, v)
        if i == target_index:
            vstart = offset + line.index(v)
            spans = [(vstart, vstart + len(v))]  # the answer-bearing value tokens
            key_span = (offset + line.index(k), offset + line.index(k) + len(k))
            record_span = (offset, offset + len(line))
        lines.append(line)
        offset += len(line) + 1
    context = "\n".join(lines)
    return Example(
        id=f"A-N{n_records}-{tier}-{position:.1f}-{idx}", task="A", context=context,
        question=f"What is the access code for {target_key}? Answer with the six-digit code only.",
        answers=[values[target_index]], spans=spans,
        meta={"tier": tier, "position": position, "n_records": n_records, "target_key": target_key,
              "key_span": key_span, "protect_spans": [record_span]},  # the whole record line
    )


def code_source(ex: Example, answer: str) -> str:
    """Where a Task A answer's code came from: the target record, another record whose key shares the target's word
    (a near-duplicate), another record with a different word, a code that is in no record, or no code at all."""
    import re
    codes = re.findall(r"\d{6}", answer)
    if not codes:
        return "no code"
    if ex.answers[0] in codes:
        return "target"
    owner = {line.rsplit(" ", 1)[1].rstrip("."): line.split(" is ")[0].removeprefix("The access code for ")
             for line in ex.context.split("\n")}.get(codes[0])
    if owner is None:
        return "not in context"
    return "same word" if owner.split("-")[0] == ex.meta["target_key"].split("-")[0] else "other word"


def synthetic_set(n_per_cell: int, n_records: int = 200, positions=(0.1, 0.5, 0.9), seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    return [synthetic_example(rng, tier, pos, n_records, i)
            for tier in TIERS for pos in positions for i in range(n_per_cell)]


# ---------- Task B: HotpotQA distractor ----------

def _paragraph(title: str, sentences: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Paragraph text and the character span of each sentence inside it."""
    head = f"{title}: "
    text, spans = head, []
    for s in sentences:
        start = len(text)
        text += s
        spans.append((start, len(text)))
    return text, spans


def _occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    """Case-insensitive, word-bounded occurrences of `needle` in `text`."""
    if needle.lower() in ("yes", "no") or not needle.strip():
        return []
    return [m.span() for m in re.finditer(rf"(?<!\w){re.escape(needle)}(?!\w)", text, flags=re.IGNORECASE)]


def hotpot_set(rows: list[dict], n_questions: int, positions=("start", "middle", "end"), seed: int = 0) -> list[Example]:
    """rows: HotpotQA distractor examples (dicts with question, answer, supporting_facts, context)."""
    rng = random.Random(seed)
    pool = [(t, s) for r in rows for t, s in zip(r["context"]["title"], r["context"]["sentences"])]
    chosen = rng.sample(rows, n_questions)
    out = []
    for qi, r in enumerate(chosen):
        titles, sents = r["context"]["title"], r["context"]["sentences"]
        gold_titles = list(dict.fromkeys(r["supporting_facts"]["title"]))
        support = {(t, i) for t, i in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"])}
        gold = [(t, s) for t, s in zip(titles, sents) if t in gold_titles]
        tfidf = [(t, s) for t, s in zip(titles, sents) if t not in gold_titles]
        for tier in ("low", "high"):
            if tier == "high":
                distractors = tfidf
            else:
                distractors = []
                while len(distractors) < len(tfidf):
                    t, s = rng.choice(pool)
                    if t not in titles and all(t != d[0] for d in distractors):
                        distractors.append((t, s))
            for pos in positions:
                order = distractors[:]
                k = {"start": 0, "middle": len(order) // 2, "end": len(order)}[pos]
                order[k:k] = gold
                text, sent_spans, ans_spans = "", [], []
                for t, s in order:
                    para, para_spans = _paragraph(t, s)
                    for i, (a, b) in enumerate(para_spans):
                        if (t, i) in support and t in gold_titles:
                            sent_spans.append((len(text) + a, len(text) + b))
                            ans_spans += [(len(text) + a + x, len(text) + a + y)
                                          for x, y in _occurrences(para[a:b], r["answer"])]
                    text += para + "\n\n"
                # The answer-bearing item is the answer string inside the supporting sentences (like Task A's value);
                # when it is not there verbatim (yes/no and paraphrased answers), fall back to the whole sentences.
                other_copy = any(not any(sa <= a and b <= sb for sa, sb in sent_spans)
                                 for a, b in _occurrences(text, r["answer"]))
                out.append(Example(
                    id=f"B-{tier}-{pos}-{qi}", task="B", context=text.strip(),
                    question=f"{r['question']} Answer with a short phrase only.",
                    answers=[r["answer"]], spans=ans_spans or sent_spans,
                    meta={"tier": tier, "position": pos, "hotpot_id": r["id"], "n_support": len(sent_spans),
                          "item": "answer" if ans_spans else "sentence", "support_spans": sent_spans,
                          "answer_elsewhere": other_copy,  # the answer string also occurs outside the support
                          "protect_spans": sent_spans},  # the supporting sentences
                ))
    return out
