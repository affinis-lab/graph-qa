from collections import defaultdict
from tqdm.auto import tqdm

from .metrics import exact_match, f1_score


class Pipeline:
    def __init__(self, retriever, reasoner, reader):
        self.retriever = retriever
        self.reasoner = reasoner
        self.reader = reader

    def __call__(self, question):
        return self.predict(question)

    def predict(self, question):
        retrieval_results = self.retriever(question)
        reasoner_results = self.reasoner(question, retrieval_results)
        reader_results = self.reader(question, reasoner_results)
        # return most probable result
        return max(reader_results, key=lambda result: result['confidence'])

    def evaluate(self, dataset):
        results = defaultdict(float)
        for instance in tqdm(dataset):
            question, gold_answer = instance
            pred_answer = self.predict(question)['answer']
            results['exact_match'] += exact_match(gold_answer, pred_answer)
            results['f1_score'] += f1_score(gold_answer, pred_answer)
        results['exact_match'] /= len(dataset)
        results['f1_score'] /= len(dataset)
        return results
    
    
class HGNPipeline:
    def __init__(self, retriever, ranker, hgn):
        self.retriever = retriever
        self.ranker = ranker
        self.hgn = hgn

    def __call__(self, question):
        return self.predict(question)

    def predict(self, question):
        retrieval_results = self.retriever(question)
        ranker_results = self.ranker(question, retrieval_results)
        hgn_results = self.hgn(question, ranker_results)
        reader_results = self.reader(question, reasoner_results)
        # return most probable result
        return max(reader_results, key=lambda result: result['confidence'])

    def evaluate(self, dataset):
        results = defaultdict(float)
        for instance in tqdm(dataset):
            question, gold_answer = instance
            pred_answer = self.predict(question)['answer']
            results['exact_match'] += exact_match(gold_answer, pred_answer)
            results['f1_score'] += f1_score(gold_answer, pred_answer)
        results['exact_match'] /= len(dataset)
        results['f1_score'] /= len(dataset)
        return results

