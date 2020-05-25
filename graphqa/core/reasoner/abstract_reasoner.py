
from abc import ABC, abstractmethod


class AbstractReasoner(ABC):
    def __call__(self, question, paragraphs):
        return self.rank(question, paragraphs)

    @abstractmethod
    def load(self, path):
        raise NotImplementedError()

    @abstractmethod
    def rank(self, question, paragraphs):
        raise NotImplementedError()
