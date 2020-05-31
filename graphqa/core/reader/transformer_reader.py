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
            args={
                'fp16': fp16,
                'n_best_size': 1,
            }
        )

    def extract_answer(self, question, paragraphs):
        to_predict = []
        contexts = {}
        for idx, entry in enumerate(paragraphs):
            if not entry:
                continue
            context = self._get_context(entry['paragraphs'])
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

        predictions, probabilities = self.model.predict(to_predict)

        results = []
        for prediction, probability in zip(predictions, probabilities):
            prediction_id = prediction['id']
            context = contexts[prediction_id]

            answer = prediction['answer'][0]
            confidence = probability['probability'][0]

            confidence += paragraphs[prediction_id]['score']
            if not answer:
                confidence = 0

            results.append({
                'answer': answer,
                'confidence': confidence,
                'context': context if answer else '',
                'supporting_facts': [],
            })

        return results

    def _get_context(self, paragraphs):
        return ' '.join(map(lambda p: p['text'], paragraphs))
