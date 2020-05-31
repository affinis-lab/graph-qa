import csv
import json
import sys
from functools import partial

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from tqdm.auto import tqdm


def load_data(path):
    with open(path) as f:
        yield from csv.DictReader(f)


def generate_actions(data_iterable):
    for entry in data_iterable:
        yield {
            "_id": entry["paragraphId:ID"],
            "text": entry["text:string"],
        }


def main(input_path, index_name):
    client = Elasticsearch()

    print(f"Creating index '{index_name}' if it doesn't exist.")
    client.indices.create(
        index=index_name,
        body={
            "settings": {"number_of_shards": 1},
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                }
            },
        },
        ignore=400,
    )
    print("Index created.")

    data = generate_actions(load_data(input_path))
    successes = 0
    with tqdm(desc='Indexing documents', unit='docs') as progress_bar:
        for ok, _ in streaming_bulk(client=client, index=index_name, actions=data):
            successes += ok
            progress_bar.update()

    print(f'Indexed {successes} documents.')

    client.indices.refresh(index=index_name)

    print('Index refreshed.')


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
