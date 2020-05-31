def recall(gold_paragraphs, pred_paragraphs):
    return sum([g_p in pred_paragraphs for g_p in gold_paragraphs]) / len(gold_paragraphs)
