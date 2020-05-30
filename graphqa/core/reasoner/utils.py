import random
import math

import numpy as np
import torch

from transformers import (  
    get_constant_schedule,
    get_linear_schedule_with_warmup, 
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup, 
    get_cosine_with_hard_restarts_schedule_with_warmup
)

from torch.utils.data import (
    TensorDataset,
    DataLoader
)


def batch_generator(sequence, batch_size=1):
    seq_length = len(sequence)
    for idx in range(0, seq_length, batch_size):
        yield sequence[idx:min(idx + batch_size, seq_length)]


def prepare_dataloader(features, batch_size = 1, shuffle = False, random_seed=42):
    # find a better way to do this globally, so that there isn't a need to pass a parameter
    random.seed(random_seed)

    if shuffle:
        random.shuffle(features)
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_masks = torch.tensor([f.input_masks for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_output_masks = torch.tensor([f.output_masks for f in features], dtype=torch.float)
    all_target_labels = torch.tensor([f.target_labels for f in features], dtype=torch.float)
    all_num_reasoning_steps = torch.tensor([f.num_reasoning_steps for f in features], dtype=torch.long)
    
    dataset = TensorDataset(
        all_input_ids,
        all_input_masks,
        all_segment_ids,
        all_output_masks,
        all_target_labels,
        all_num_reasoning_steps
    )

    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    
    return dataloader

def shuffle_features(input_ids, segment_ids, input_masks, target_labels, max_paragraph_len, golden_indices, output_masks=None):
    permutation = torch.randperm(max_paragraph_len)

    p_input_ids = input_ids[:, permutation]
    p_segment_ids = segment_ids[:, permutation]
    p_input_masks = input_masks[:, permutation]
    
    # dont permute the last paragraph because it represents EOE (max_paragraph_len is the index of EOE)
    permutation_target = torch.cat((permutation, torch.LongTensor([max_paragraph_len])))
    
    p_target_labels = target_labels[:, :, permutation_target]
    
    # regain the indices of golden paragraphs after the permutation
    sorter = np.argsort(permutation)
    golden_indices = sorter[np.searchsorted(permutation, golden_indices, sorter=sorter)].tolist() 
    
    if output_masks is not None:
        p_output_masks = output_masks[:, :, permutation_target]
        return p_input_ids, p_segment_ids, p_input_masks, p_target_labels, p_output_masks, golden_indices
    
    return p_input_ids, p_segment_ids, p_input_masks, p_target_labels, golden_indices

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + math.cos(math.pi * x))

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

class InputFeatures(object):

    def __init__(self, input_ids, input_masks, segment_ids, output_masks, target_labels, num_paragraphs, num_reasoning_steps):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.output_masks = output_masks
        self.target_labels = target_labels
        self.num_paragraphs = num_paragraphs
        self.num_reasoning_steps = num_reasoning_steps
        
    def get_input_values(self):
        return self.input_ids, self.input_masks, self.segment_ids, self.output_masks, self.target_labels
