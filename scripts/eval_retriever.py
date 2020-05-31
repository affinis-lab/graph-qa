import argparse
import json
from functools import partial
from itertools import chain
from multiprocessing import Manager, Pool

from tqdm.auto import tqdm

from graphqa.core.retriever import TfIdfGraphRetriever, ElasticGraphRetriever
from graphqa.core.retriever.utils import recall


model = None


def get_model(model_name, model_path, *args, **kwargs):
    models = {
        'gensim': TfIdfGraphRetriever,
        'elastic': ElasticGraphRetriever,
    }

    model = models[model_name](*args, **kwargs)
    model.load(model_path)
    return model


def load_dataset(name, path):
    datasets = {
        'squad': load_squad,
        'hotpotqa': load_hotpotqa,
    }
    return datasets[name](path)


def load_squad(path):
    # TODO
    pass


def load_hotpotqa(path):
    with open(path) as f:
        dataset = json.load(f)

    def extract_golds(titles, context):
        golds = []
        for paragraph in context:
            if paragraph[0] in titles:
                golds.append(''.join(paragraph[1]))
        return golds

    eval_dataset = []
    for entry in dataset:
        question = entry['question']
        gold_titles = list(map(lambda x: x[0], entry['supporting_facts']))
        gold_paragraphs = extract_golds(gold_titles, entry['context'])
        eval_dataset.append((question, gold_paragraphs))
    return eval_dataset

def init_model(*args, **kwargs):
    global model
    model = get_model(*args, **kwargs)


def evaluate(args):
    question, gold_paragraphs, output = args
    pred_paragraphs = [p['text'] for p in chain.from_iterable(model.retrieve(question))]
    output.append(recall(gold_paragraphs, pred_paragraphs))


def main(dataset_name, dataset_path, *args, **kwargs):
    dataset = load_dataset(dataset_name, dataset_path)

    with Manager() as manager:
        recall = manager.list()
        init_model_func = partial(init_model, *args, **kwargs)
        with tqdm(desc='Evaluating', unit='paragraphs', total=len(dataset)) as progress_bar:
            with Pool(initializer=init_model_func) as pool:
                get_args = lambda x: x + (recall,)
                for _ in pool.imap(evaluate, map(get_args, dataset)):
                    average_recall = sum(recall) / len(recall)
                    progress_bar.set_description(desc=f'Recall: {average_recall:.2f}')
                    progress_bar.update()

        average_recall = sum(recall) / len(recall)
        print(f'Final recall: {average_recall}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retriever evaluation script.')
    subparsers = parser.add_subparsers(help='Arguments for models', dest='model_name')

    parser.add_argument(
        'dataset_name',
        metavar='Dataset name',
        type=str,
        help="Dataset name determines what loader to use.",
        choices=['squad', 'hotpotqa'],
    )

    parser.add_argument(
        'dataset_path',
        metavar='Dataset path',
        type=str,
        help="Path from which to load dataset.",
    )

    parser.add_argument(
        '--model_path',
        metavar='Model path',
        type=str,
        nargs='?',
        help="Path from which to load model.",
        default=None,
    )

    parser_tfidf = subparsers.add_parser('tfidf')
    parser_tfidf.add_argument(
        '--graph-db-address',
        metavar='Graph DB address',
        type=str,
        nargs='?',
        help="Graph DB address",
        default='bolt://localhost:7687',
        dest='db_addr'
    )

    parser_tfidf.add_argument(
        '--num-best',
        metavar='Number of matching paragraphs',
        type=int,
        nargs='?',
        help="Number of initial paragraphs to retrieve",
        default=10,
    )

    parser_tfidf.add_argument(
        '--num-related',
        metavar='Number of related paragraphs',
        type=int,
        nargs='?',
        help="Number of connected paragraphs to retrieve for multihop.",
        default=50,
    )

    parser_tfidf.add_argument(
        '--max-path-depth',
        metavar='Max path depth',
        type=int,
        nargs='?',
        help="Maximal distance between nodes in Graph DB.",
        default=2,
    )

    parser_elastic = subparsers.add_parser('elastic')
    parser_elastic.add_argument(
        '--elastic-address',
        metavar='Elasticsearch address',
        type=str,
        nargs='?',
        help="Elasticsearch address",
        default='localhost:9200',
        dest='es_addr'
    )

    parser_elastic.add_argument(
        '--graph-db-address',
        metavar='Graph DB address',
        type=str,
        nargs='?',
        help="Graph DB address",
        default='bolt://localhost:7687',
        dest='db_addr'
    )

    parser_elastic.add_argument(
        '--index-name',
        metavar='Index name',
        type=str,
        nargs='?',
        help="Name of the index in Elasticsearch DB.",
        default='paragraphs',
    )

    parser_elastic.add_argument(
        '--num-best',
        metavar='Number of matching paragraphs',
        type=int,
        nargs='?',
        help="Number of initial paragraphs to retrieve",
        default=10,
    )

    parser_elastic.add_argument(
        '--num-related',
        metavar='Number of related paragraphs',
        type=int,
        nargs='?',
        help="Number of connected paragraphs to retrieve for multihop.",
        default=50,
    )

    parser_elastic.add_argument(
        '--max-path-depth',
        metavar='Max path depth',
        type=int,
        nargs='?',
        help="Maximal distance between nodes in Graph DB.",
        default=2,
    )

    args = vars(parser.parse_args())
    main(**args)
