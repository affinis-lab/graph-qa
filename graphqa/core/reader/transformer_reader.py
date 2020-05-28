import torch
from simpletransformers.question_answering import QuestionAnsweringModel

from .abstract_reader import AbstractReader


class TransformerReader(AbstractReader):

    def __init__(self):
        super().__init__()
        self.model = None

    def load(self, path):
        use_cuda = torch.cuda.is_available()

        if not use_cuda:
            raise UserWarning('CUDA not available.')

        try:
            import apex
            fp16 = True
        except ImportError:
            fp16 = False

        self.model = QuestionAnsweringModel(
            model_type='albert',
            model_name=path,
            use_cuda=use_cuda,
            args={'fp16': fp16}
        )

    def extract_answer(self, question, paragraphs):
        to_predict = []
        contexts = {}
        for idx, entry in enumerate(paragraphs):
            if not entry:
                continue
            context = self._get_context(entry)
            contexts[idx] = context
            to_predict.append({
                'context': context,
                'qas': [{'question': question, 'id': idx}]
            })

        if not to_predict:
            return [{
                'answer': '',
                'context': '',
                'confidence': 0,
                'supporting_facts': []
            }]

        predictions = self.model.predict(to_predict)
        results = []
        for prediction in predictions:
            idx = prediction['id']
            context = contexts[idx]
            answer = prediction['answer']
            results.append({
                'answer': answer,
                'confidence': 1.0 if answer else 0.5,
                'context': context,
                'supporting_facts': [],
            })
        return results

    def _get_context(self, paragraphs):
        return ' '.join(map(lambda p: p['text'], paragraphs))
