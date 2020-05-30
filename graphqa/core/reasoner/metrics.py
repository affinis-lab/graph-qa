import numpy as np


# position in a reasoning path doesen't matter, it is only important to get the valid paragraph
def intersection_ratio_metric(output_indexes, target_indexes):
    assert len(output_indexes) == len(target_indexes)
    
    return len(set(output_indexes).intersection(target_indexes)) / len(output_indexes)

# order matters
def exact_match_metric(output_indexes, target_indexes, reasoning_path_len=None):
    if not reasoning_path_len:
        reasoning_path_len = len(target_indexes)

    num_matching_paragraphs = np.sum(output_indexes == target_indexes)
    return num_matching_paragraphs / reasoning_path_len
