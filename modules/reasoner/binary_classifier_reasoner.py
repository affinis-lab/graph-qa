import numpy as np

import torch
from simpletransformers.classification import ClassificationModel

from ..utils import softmax
from constants import REASONER_MODEL_PATH

class BinaryClassifierReasoner:

    def __init__(self):
        self.load_model()

    def __call__(self, *args, **kwargs):
        question = kwargs['question']
        paragraphs = kwargs['paragraphs']
        question, paragraphs = self.rank(question, paragraphs)
        return [], {'question': question, 'paragraphs': paragraphs}

    def load_model(self):
        use_cuda = torch.cuda.is_available()

        if not use_cuda:
            print('Warning: Not using CUDA!')

        train_args = {
            'fp16': False,
            'reprocess_input_data': True,
            'overwrite_output_dir': True,
            'evaluate_during_training': False,
            'evaluate_during_training_steps': False,
            'max_seq_length': 192,
            'num_train_epochs': 5,
            'train_batch_size': 16,
            'eval_batch_size': 16,
            'no_cache':True
        }

        self.model = ClassificationModel('albert', REASONER_MODEL_PATH, num_labels=2, use_cuda=use_cuda, args=train_args)
        
    def rank(self, question, paragraphs):
        model_input = self._prepare_model_input(question, paragraphs)
        
        predictions, raw_outputs = self.model.predict(model_input)        
        softmax_outputs = [list(softmax(logits, theta=0.3)) for logits in raw_outputs]

        paragraphs = self._determine_best_paragraphs(model_input, softmax_outputs)
        
        print(predictions)
        print(softmax_outputs)
        print(paragraphs)

        return question, paragraphs

    def _prepare_model_input(self, question, paragraphs):
        return [[question, paragraph] for paragraph in paragraphs]
    
    def argsort(self, l):
        return sorted(range(len(l)), key=l.__getitem__)

    def _determine_best_paragraphs(self, model_input, softmax_outputs):
        indexes = self.argsort(softmax_outputs)
        print(indexes)
        # limit to 2 paragraphs because of the HotpotQA setting
        return [model_input[i][1] for i in indexes][:2]