import json
import os
import logging

from typing import Dict, List
from argparse import ArgumentParser
from bert_serving.client import BertClient

logging.basicConfig(level=logging.INFO)


def create_document(paragraph: Dict,
                    embedding: any,
                    index_name: str):
    return {
        "_op_type": "index",
        "_index": index_name,
        "id": paragraph["id"],
        "topic_name": paragraph["topic_name"],
        "topic_text": paragraph["topic_text"],
        "topic_text_vector": embedding
    }


def bulk_predict(paragraphs: List[Dict],
                 bert_client: BertClient,
                 batch_size=256):
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i: i+batch_size]
        embeddings = bert_client.encode(
            [paragraph["topic_text"] for paragraph in batch]
        )
        for embedding in embeddings:
            yield embedding


def get_parsed_paragraphs(load_path: str) -> Dict:
    with open(load_path, 'r') as fp:
        data = json.load(fp)
        return data


def create_list_of_paragraphs(topic: Dict) -> List[Dict]:
    list_of_paragraphs = []
    # for topic_id in data:
    #     topic = data[topic_id]
    topic_name = topic["topic_name"]
    paragraphs = topic["paragraphs"]
    for paragraph_id in paragraphs:
        paragraph = paragraphs[paragraph_id]
        paragraph_text = paragraph["text"]
        item = dict()
        item["id"] = paragraph_id
        item["topic_name"] = topic_name
        item["topic_text"] = paragraph_text
        list_of_paragraphs.append(item)
    return list_of_paragraphs


def create_list_of_sentences(topic: Dict) -> List[Dict]:
    list_of_sentence = []
    topic_name = topic["topic_name"]
    paragraphs = topic["paragraphs"]
    for paragraph_id in paragraphs:
        paragraph = paragraphs[paragraph_id]
        # paragraph_text = paragraph["text"]
        paragraph_sentences = paragraph["sentences"]
        sentence_counter = 0
        for sentence in paragraph_sentences:
            sentence_counter += 1
            sentence_id = paragraph_id + "-" + str(sentence_counter)
            item = dict()
            item["id"] = sentence_id
            item["topic_name"] = topic_name
            item["topic_text"] = sentence
            list_of_sentence.append(item)
    return list_of_sentence


def main(arguments):
    bc = BertClient(output_fmt='list', check_length=False)
    logging.info("start")
    index_name = arguments.index
    json_path = arguments.json
    save_path = arguments.output
    for topic_name in os.listdir(json_path):
        load_path = os.path.join(json_path, topic_name)
        data = get_parsed_paragraphs(load_path)
        logging.info("done parsing paragraphs. [1/2]")
        list_of_paragraphs = create_list_of_sentences(data)
        logging.info("done creating list of paragraphs. [2/2]")
        with open(save_path, 'a') as f:
            counter = 0
            for paragraph, embedding in \
                    zip(list_of_paragraphs,
                        bulk_predict(list_of_paragraphs,
                                     bc)):
                counter += 1
                logging.info("counter value is: ", counter)
                logging.info("paragraph id: ", paragraph["id"])
                d = create_document(paragraph, embedding, index_name)
                f.write(json.dumps(d) + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--index', required=True, help='name of the ES index (es_sentences)')
    parser.add_argument('--json', required=True, help='path to the directory with input json files')
    parser.add_argument('--output', required=True, help='name of the output file (output_sentences.json1)')
    args = parser.parse_args()

    main(args)
