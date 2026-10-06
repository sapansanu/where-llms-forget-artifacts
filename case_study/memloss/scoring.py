"""Answer scoring. Task A: the six-digit code must appear in the answer. Task B: the official HotpotQA normalisation
for exact match and token F1."""

import re
import string
from collections import Counter


def code_correct(answer: str, gold: str) -> bool:
    return gold in re.findall(r"\d{6}", answer)


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(answer: str, gold: str) -> bool:
    return normalize(answer) == normalize(gold)


def f1(answer: str, gold: str) -> float:
    pred, ref = normalize(answer).split(), normalize(gold).split()
    if normalize(answer) in ("yes", "no", "noanswer") or normalize(gold) in ("yes", "no", "noanswer"):
        return float(normalize(answer) == normalize(gold))
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return 2 * precision * recall / (precision + recall)
