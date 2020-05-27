from .abstract_reasoner import AbstractReasoner


class DummyReasoner(AbstractReasoner):

    def load(self, path):
        pass

    def rank(self, question, paragraphs):
        if not paragraphs:
            return []
        if type(paragraphs[0]) != list:
            return paragraphs[:2]
        return [p[:2] for p in paragraphs]
