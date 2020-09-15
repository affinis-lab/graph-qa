import logging

from json import load
from argparse import ArgumentParser
from elasticsearch import Elasticsearch

logging.basicConfig(level=logging.INFO)


INDEX_NAME = "es_sentences"
CONFIG_PATH = "index_config.json"


def create_index(es: Elasticsearch,
                 index_name: str,
                 config_path: str) -> None:
    try:
        with open(config_path) as file:
            config = load(file)

        es.indices.create(index=index_name, body=config)
        logging.info("index " + index_name + " has been created!")
    except:
        logging.warning("some exception has occurred!")


def main(arguments):
    es = Elasticsearch('localhost:9200')

    index_name = arguments.index
    config_path = arguments.config

    create_index(es,
                 index_name,
                 config_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--index', required=True, help='name of the ES index (es_sentences)')
    parser.add_argument('--config', required=True, help='path to the config file (index_config.json)')
    args = parser.parse_args()

    main(args)
