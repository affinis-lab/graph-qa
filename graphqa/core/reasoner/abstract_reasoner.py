
from abc import ABC, abstractmethod


class AbstractReasoner(ABC):
    @abstractmethod
    def load(self, path):
        raise NotImplementedError()

    @abstractmethod
    def rank(self, question, paragraphs):
        raise NotImplementedError()
