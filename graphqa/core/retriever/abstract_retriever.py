from abc import ABC, abstractmethod


class AbstractRetriever(ABC):
    def __call__(self, question):
        return self.retrieve(question)

    @abstractmethod
    def load(self, path):
        raise NotImplementedError()

    @abstractmethod
    def retrieve(self, question):
        raise NotImplementedError()
