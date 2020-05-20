from abc import ABC, abstractmethod


class AbstractRetriever(ABC):
    @abstractmethod
    def load(self, path):
        raise NotImplementedError()

    @abstractmethod
    def retrieve(self, question, paragraphs):
        raise NotImplementedError()
