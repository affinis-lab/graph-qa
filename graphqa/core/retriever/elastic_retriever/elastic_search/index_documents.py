import json
from typing import List
from argparse import ArgumentParser

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


def load_data_set(path: str) -> List:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main(arguments):
    client = Elasticsearch('localhost:9200')
    docs = load_data_set(arguments.data)
    bulk(client, docs)


if __name__ == '__main__':
    parser = ArgumentParser(description='indexing ES documents.')
    parser.add_argument('--data', help='ES documents (output_sentences.json1)')
    args = parser.parse_args()
    main(args)
