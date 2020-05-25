class Pipeline:
    def __init__(self, retriever, reasoner, reader):
        self.retriever = retriever
        self.reasoner = reasoner
        self.reader = reader

    def __call__(self, question):
        retrieval_results = self.retriever(question)
        reasoner_results = self.reasoner(question, retrieval_results)
        reader_results = self.reader(question, reasoner_results)
        # return most probable result
        return max(reader_results, key=lambda result: result['confidence'])
