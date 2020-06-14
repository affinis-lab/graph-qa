from abc import ABC, abstractmethod


class AbstractRanker(ABC):

    @abstractmethod
    def load(self, path):
        raise NotImplementedError

    @abstractmethod
    def rank(self, question, paragraphs):
        raise NotImplementedError

    @abstractmethod
    def score(self, question, paragraph):
        raise NotImplementedError
