from modules.reader.abstract_reader import AbstractReader

from simpletransformers.question_answering import QuestionAnsweringModel

import json
import os


class BertReader(AbstractReader):

    def __init__(self, model_path):
        super().__init__()

        self.model_path = model_path
        self.load_model()

    def __call__(self, *args, **kwargs):
        question = kwargs['question']
        paragraphs = kwargs['paragraphs']
        answer = self.extract_answer(question, paragraphs)
        return [], {'answer': answer} 

    def load_model(self):
        train_args = {
            'fp16': False,
            'num_train_epochs': 3,
            'max_seq_length': 192,
            'doc_stride': 128,
            'overwrite_output_dir': True,
            'reprocess_input_data': False,
            'train_batch_size': 8,
            'eval_batch_size': 8,
            'gradient_accumulation_steps': 8,
            'no_cache': True
        }
        self.model = QuestionAnsweringModel('albert', self.model_path, args=train_args)
        
    def extract_answer(self, question, paragraphs):
        print("Your question is: ", question)
        
        print('Looking for an answer in ', paragraphs[0])
        to_predict = [
            {'context': paragraphs[0], 'qas': [{'question': question, 'id': '0'}]}
        ]

        return self.model.predict(to_predict)