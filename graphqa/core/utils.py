import collections
import string
import re

import nltk


def tokenize(text, stopwords=[]):
    tokens = nltk.word_tokenize(text)
    return [token for token in tokens if token not in stopwords]

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def exact_match(gold_answer, pred_answer):
    return int(normalize_answer(gold_answer) == normalize_answer(pred_answer))

def f1_score(gold_answer, pred_answer):
    gold_tokens = get_tokens(gold_answer)
    pred_tokens = get_tokens(pred_answer)
    common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
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
