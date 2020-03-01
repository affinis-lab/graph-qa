import bz2, json, re
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.retriever.abstract_retriever import AbstractRetriever
from utils.corpus_loader import CorpusLoader
from constants import RETRIEVER_VECTOR_REPRESENTATION_PATH


class TfIdfRetriever(AbstractRetriever):

    def __init__(self):
        super().__init__()

        self.corpus_loader = CorpusLoader()
        self.corpus = self.corpus_loader.load_corpus()

    def __call__(self, *args, **kwargs):
        question = kwargs['question']
        paragraphs = kwargs['paragraphs']
        question, paragraphs = self.retrieve(question, paragraphs)
        return [], {'question': question,  'paragraphs': paragraphs}

    def retrieve(self, question, paragraphs):
        vectorizer = TfidfVectorizer(ngram_range=(2, 4), stop_words=['english'], preprocessor=self.corpus_loader.preprocess)
        # vector_representation = self.load_vector_representation()
        vector_representation = vectorizer.fit_transform(self.corpus)

        similarities = self.get_tf_idf_query_similarity(vectorizer, vector_representation, question)
        paragraphs.extend(self.retrieve_best_paragraphs(similarities, n_best=10))

        pretty_print(paragraphs)

        return question, paragraphs

    def get_tf_idf_query_similarity(self, vectorizer, docs_tfidf, query):
        query_tfidf = vectorizer.transform([self.corpus_loader.preprocess(query).lower()])
        return cosine_similarity(query_tfidf, docs_tfidf).flatten()

    def retrieve_best_paragraphs(self, similarities, n_best=5):
        best_paragraphs = similarities.argsort()[-n_best:][::-1].tolist()
        return [self.corpus[index] for index in best_paragraphs]


def pretty_print(ranked_paragraphs):
    for i, text in enumerate(ranked_paragraphs):
        print(i+1, text, '\n')