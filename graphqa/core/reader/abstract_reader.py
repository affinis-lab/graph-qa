from abc import ABC, abstractmethod


class AbstractReader(ABC):
    @abstractmethod
    def load(self, path):
        raise NotImplementedError()

    @abstractmethod
    def extract_answer(self, question, paragraphs):
        raise NotImplementedError()
