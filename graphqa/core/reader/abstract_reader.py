from abc import ABC, abstractmethod


class AbstractReader(ABC):
    def __call__(self, question, paragraphs):
        return self.extract_answer(question, paragraphs)

    @abstractmethod
    def load(self, path):
        raise NotImplementedError()

    @abstractmethod
    def extract_answer(self, question, paragraphs):
        raise NotImplementedError()
