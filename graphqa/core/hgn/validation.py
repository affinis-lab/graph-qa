from collections import defaultdict, Counter
from itertools import chain

from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import classification_report

import torch
import torch.functional as F

from .utils import get_best_indexes
from ..utils import get_tokens

PARAGRAPH_THRESHOLD = 0.5
SENTENCE_THRESHOLD = 0.3


def f1_score_qa(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_score(target_labels, pred_labels):
    num_same = (pred_labels == target_labels).sum()

    precision = 1.0 * num_same / len(pred_labels)
    recall = 1.0 * num_same / len(target_labels)

    if precision == 0 and recall == 0:
        return 0

    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def validate_qa(input_ids, start_logits, end_logits, start_positions, end_positions, tokenizer):
    start_indexes = get_best_indexes(start_logits.flatten())[0]
    end_indexes = get_best_indexes(end_logits.flatten())[0]

    input_ids = input_ids[0]

    start_positions = start_positions[0].tolist()[0]
    end_positions = end_positions[0].tolist()[0]

    answer = tokenizer.decode(input_ids[start_indexes:end_indexes+1])
    true_answer = tokenizer.decode(input_ids[start_positions:end_positions+1])

    return f1_score_qa(true_answer, answer)


def validate(hgn, validation_dataloader, device, num_validation_steps=None, gradient_accumulation_steps=1):
    if not num_validation_steps or num_validation_steps > len(validation_dataloader):
        num_validation_steps = len(validation_dataloader)

    model = hgn.model
    tokenizer = hgn.tokenizer

    validation_metrics = defaultdict(float)

    all_labels = {
        'paragraphs': list(),
        'sentences': list()
    }
    all_predictions = {
        'paragraphs': list(),
        'sentences': list()
    }

    model.eval()
    progress_bar = tqdm(validation_dataloader, desc='Valid', position=0, leave=True)
    with torch.no_grad():
        for step, batch in enumerate(progress_bar):
            if step == num_validation_steps:
                break
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_masks, segment_ids, labels, edge_indexes, node_offsets, paragraph_node_indexes, sentence_node_indexes, start_positions, end_positions  = batch

            paragraph_preds, sentence_preds, start_logits, end_logits = model(
                        input_ids,
                        input_masks,
                        segment_ids,
                        node_offsets,
                        paragraph_node_indexes,
                        sentence_node_indexes,
                        edge_indexes=edge_indexes)

            # TODO: retain batch dim
            labels = labels.view(labels.size(1))

            paragraph_labels = labels[paragraph_node_indexes[paragraph_node_indexes != -1]].unsqueeze(1)
            sentence_labels = labels[sentence_node_indexes[sentence_node_indexes != -1]].unsqueeze(1)

            paragraph_preds = torch.sigmoid(paragraph_preds)
            sentence_preds = torch.sigmoid(sentence_preds)

            paragraph_loss = F.binary_cross_entropy_with_logits(paragraph_preds, paragraph_labels, reduction='mean')
            sentence_loss = F.binary_cross_entropy_with_logits(sentence_preds, sentence_labels, reduction='mean')

            validation_metrics['paragraphs_loss'] += paragraph_loss.item() / gradient_accumulation_steps
            validation_metrics['sentences_loss'] += sentence_loss.item() / gradient_accumulation_steps

            paragraph_preds[paragraph_preds <= PARAGRAPH_THRESHOLD] = 0
            paragraph_preds[paragraph_preds > PARAGRAPH_THRESHOLD] = 1

            sentence_preds[sentence_preds <= SENTENCE_THRESHOLD] = 0
            sentence_preds[sentence_preds > SENTENCE_THRESHOLD] = 1

            # TODO: extend instead of append
            all_labels['paragraphs'].append(paragraph_labels.flatten().tolist())
            all_labels['sentences'].append(sentence_labels.flatten().tolist())

            all_predictions['paragraphs'].append(paragraph_preds.flatten().tolist())
            all_predictions['sentences'].append(sentence_preds.flatten().tolist())

            paragraphs_f1_score = f1_score(paragraph_labels, paragraph_preds)
            sentences_f1_score = f1_score(sentence_labels, sentence_preds)

            validation_metrics['paragraphs_f1_score'] += paragraphs_f1_score
            validation_metrics['sentences_f1_score'] += sentences_f1_score

            # calculate QA f1 score
            qa_f1_score = validate_qa(input_ids, start_logits, end_logits, start_positions, end_positions, tokenizer)
            validation_metrics['qa_f1_score'] += qa_f1_score

            progress_bar.set_description("Step %i Para: F1 %.2f Sent: F1 %.2f QA: F1 %.2f"
                                             % (int(step+1),
                                                float(validation_metrics['paragraphs_f1_score']/(step+1)),
                                                float(validation_metrics['sentences_f1_score']/(step+1)),
                                                float(validation_metrics['qa_f1_score']/(step+1))))

    validation_metrics['paragraphs_loss'] /= num_validation_steps
    validation_metrics['sentences_loss'] /= num_validation_steps

    validation_metrics['paragraphs_f1_score'] /= num_validation_steps
    validation_metrics['sentences_f1_score'] /= num_validation_steps

    validation_metrics['qa_f1_score'] /= num_validation_steps

    print('#'*50)
    print('QA f1 score: ', validation_metrics['qa_f1_score'])
    print('Paragraphs loss: ', validation_metrics['paragraphs_loss'])
    print('Sentences loss: ', validation_metrics['sentences_loss'])
    print('Paragraphs results:')
    print(classification_report(np.array(list(chain.from_iterable(all_labels['paragraphs']))), list(chain.from_iterable(np.array(all_predictions['paragraphs'])))))
    print('Sentences results:')
    print(classification_report(np.array(list(chain.from_iterable(all_labels['sentences']))), list(chain.from_iterable(np.array(all_predictions['sentences'])))))
    print('#'*50)

    return validation_metrics
