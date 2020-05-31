from functools import partial

from elasticsearch import Elasticsearch
from py2neo import Graph

from .abstract_retriever import AbstractRetriever


class ElasticGraphRetriever(AbstractRetriever):

    def __init__(self, db_addr, es_addr, index_name, num_best=5, num_related=50, max_path_depth=2):
        super().__init__()
        self.graph_db =  Graph(db_addr)
        self.elasticsearch = Elasticsearch(es_addr)
        self.index_name = index_name
        self.num_best = num_best
        self.num_related = num_related
        self.max_path_depth = max_path_depth


    def load(self, path):
        # This model has no loadable components.
        pass

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
        request = {
            "from" : 0,
            "size" : self.num_best,
            "query": {
                "match": {
                    "text": {
                        "query": question,
                        "fuzziness": "AUTO"
                    }
                }
            }
        }

        try:
            results = self.elasticsearch.search(index=self.index_name, body=request, request_timeout=30)
        except:
            return []

        paragraph_ids = []
        for hit in results["hits"]["hits"]:
            paragraph_ids.append((hit["_id"], hit["_score"]))

        return paragraph_ids

    def _fetch_related(self, paragraph_id, num_related, max_path_depth):
        query = f"match p=(a:Paragraph {{ paragraphId: \"{ paragraph_id }\" }})-" \
            f"[*..{max_path_depth}]->(b:Paragraph) " \
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
