from functools import partial
from pathlib import Path

import nltk
from gensim.corpora import Dictionary
from gensim.similarities import Similarity
from nltk.corpus import stopwords
from py2neo import Graph

from .abstract_retriever import AbstractRetriever


class TfIdfGraphRetriever(AbstractRetriever):

    def __init__(self, db, tokenizer, num_best=5, num_related=9, max_path_depth=2):
        super().__init__()
        self.tokenizer = tokenizer
        self.graph_db =  Graph(db)
        self.num_best = num_best
        self.num_related = num_related
        self.max_path_depth = max_path_depth
        self.paragraph_ids = []
        self.dictionary = None
        self.index = None

    def load(self, path):
        if type(path) == str:
            path = Path(path)

        with open(path / 'paragraph-ids.txt') as f:
            self.paragraph_ids = [paragraph_id.strip() for paragraph_id in f]

        dictionary_path = str(path / 'dct.pkl')
        self.dictionary = Dictionary.load(dictionary_path)

        index_path = str(path / 'indexes' / 'master-index')
        self.index = Similarity.load(index_path)
        self.index.num_best = self.num_best

    def retrieve(self, question):
        matched_paragraph_ids = self._match_paragraphs(question)

        fetch_related = partial(
            self._fetch_related,
            num_related=self.num_related,
            max_path_depth=self.max_path_depth,
        )

        results = []
        for paragraph_id, score in matched_paragraph_ids:
            paragraphs = fetch_related(paragraph_id)
            results.append(paragraphs)

        return results

    def _match_paragraphs(self, question):
        query = self.dictionary.doc2bow(self.tokenizer(question))
        results = []
        for idx, score in self.index[query]:
            results.append((self.paragraph_ids[idx], score))
        return results

    def _fetch_related(self, paragraph_id, num_related, max_path_depth):
        query = f"match p=(a:Paragraph {{ paragraphId: \"{ paragraph_id }\" }})-" \
            f"[*..{max_path_depth}]-(b:Paragraph) " \
            "return a,b " \
            "order by length(p) " \
            f"limit {num_related}"
        result = self.graph_db.run(query)
        try:
            data = result.to_data_frame()
            nodes = set(data.values.flatten())
            return list(map(dict, nodes))
        except KeyError:
            return None
