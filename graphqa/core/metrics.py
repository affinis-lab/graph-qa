from collections import Counter

from .utils import normalize_answer, get_tokens


def exact_match(gold_answer, pred_answer):
    return int(normalize_answer(gold_answer) == normalize_answer(pred_answer))

def f1_score(gold_answer, pred_answer):
    gold_tokens = get_tokens(gold_answer)
    pred_tokens = get_tokens(pred_answer)
    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        # if either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_tokens == pred_tokens)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
