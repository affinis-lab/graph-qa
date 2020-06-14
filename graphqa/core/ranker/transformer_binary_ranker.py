import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .abstract_ranker import AbstractRanker


class TransformerBinaryRanker(AbstractRanker):

    def __init__(self, use_gpu=True):
        self.tokenizer = None
        self.model = None

        self.use_gpu = torch.cuda.is_available() and use_gpu

    def load(self, path):
        # TODO: parametrize tokenizer loading
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained(path)

        if self.use_gpu:
            self.model = self.model.to('cuda')

    def rank(self, question, paragraphs):
        if not self.model:
            raise RuntimeError('TransformerBinaryRanker model not loaded.')

        selected_paragraphs = []
        for paragraph_group in paragraphs:
            selected = []
            for paragraph in map(lambda x: x['text'], paragraph_group):
                prediction, confidence = self.score(question, paragraph)
                if prediction == 1:
                    selected.append(paragraph)
            selected_paragraphs.append(selected)
        return selected_paragraphs

    def score(self, question, text):
        encoded_features = dict(self.tokenizer.encode_plus(
            question,
            text,
            add_special_tokens=True,
            max_length=self.tokenizer.model_max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            return_attention_mask=True,
        ))

        if self.use_gpu:
            # move all tensors to gpu
            for k, v in encoded_features.items():
                encoded_features[k] = v.to('cuda')

        logits = self.model(**encoded_features)[0][0]

        probabilities = F.softmax(logits)
        if self.use_gpu:
            probabilities = probabilities.to('cpu')
        prediction = torch.argmax(probabilities)
        confidence = probabilities[prediction]

        return prediction.item(), confidence.item()
