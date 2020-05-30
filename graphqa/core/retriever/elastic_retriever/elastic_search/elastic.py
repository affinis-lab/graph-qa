import logging
from typing import Dict, List

from bert_serving.client import BertClient
from elasticsearch import Elasticsearch, NotFoundError

from graphqa.core import AbstractRetriever

logging.basicConfig(level=logging.INFO)


class ElasticRetriever(AbstractRetriever):

    def __init__(self,
                 total_number=14,
                 index_name='es_sentences',
                 ip_address='localhost:9200'):
        super().__init__()
        self.total_number = total_number
        self.index_name = index_name
        self.paragraph_ids = []
        self.ip_address = ip_address

    def load(self, path):
        self.ip_address = path

    def retrieve(self, question) -> List:
        # establishing connections
        bc = BertClient(ip='localhost', output_fmt='list', check_length=False)
        client = Elasticsearch(self.ip_address)

        query_vector = bc.encode([question])[0]

        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, doc['topic_text_vector']) + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
        results = []
        try:
            response = client.search(
                index=self.index_name,  # name of the index
                body={
                    "size": self.total_number,
                    "query": script_query,
                    "_source": {"includes": ["id", "topic_name", "topic_text"]}
                }
            )
            logging.info(response)
            results = self.post_process_response(response)
        except ConnectionError:
            logging.warning("docker isn't up and running!")
        except NotFoundError:
            logging.warning("no such index!")
        return results

    def post_process_response(self,
                              response: Dict) -> List:
        scored_responses = response["hits"]["hits"]
        processed_response = dict()
        target_sentences = []
        for score_object in scored_responses:
            score = score_object["_score"]
            source = score_object["_source"]
            sentence_id = source["id"]
            tokenized_sentence_id = sentence_id.split("-")
            topic_id = tokenized_sentence_id[0]
            topic_name = source["topic_name"]
            sentence = source["topic_text"]
            target_sentences.append(sentence)
            if topic_id not in processed_response:
                processed_response[topic_id] = dict()
                processed_response[topic_id]["count"] = 0
                processed_response[topic_id]["topic_name"] = topic_name
                processed_response[topic_id]["sum_score"] = 0
                processed_response[topic_id]["sentence_ids"] = []
            processed_response[topic_id]["count"] += 1
            processed_response[topic_id]["sum_score"] += score
            processed_response[topic_id]["sentence_ids"].append(sentence_id)
        logging.info(processed_response)
        ranking_dictionary = dict()
        for topic_id in processed_response:
            topic = processed_response[topic_id]
            count = topic["count"]
            sum_score = topic["sum_score"]
            topic_name = topic["topic_name"]
            if count not in ranking_dictionary:
                ranking_dictionary[count] = dict()
            average_score = sum_score / count
            ranking_dictionary[count][topic_name] = average_score

        logging.info(ranking_dictionary)

        return target_sentences
