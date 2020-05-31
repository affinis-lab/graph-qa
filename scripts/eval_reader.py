import argparse
import json
from functools import partial
from multiprocessing import Manager, Pool

from tqdm.auto import tqdm

from graphqa.core.reader import TransformerReader
from graphqa.core.metrics import exact_match, f1_score


def get_reader(model_name, model_path, *args, **kwargs):
    models = {
        'transformer': TransformerReader,
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
                golds.append({'text': ''.join(paragraph[1])})
        return golds

    eval_dataset = []
    for entry in dataset:
        question = entry['question']
        answer = entry['answer']
        gold_titles = list(map(lambda x: x[0], entry['supporting_facts']))

        paragraphs = [{
            'paragraphs': extract_golds(gold_titles, entry['context']),
            'score': 1.0
        }]

        eval_dataset.append((question, answer, paragraphs))
    return eval_dataset


def main(dataset_name, dataset_path, model_name, model_path, *args, **kwargs):
    dataset = load_dataset(dataset_name, dataset_path)
    model = get_reader(model_name, model_path, *args, **kwargs)
    em_metric, f1_metric = [], []
    with tqdm(desc='Evaluating', unit='p', total=len(dataset), dynamic_ncols=True) as progress_bar:
        for question, gold_answer, gold_paragraphs in dataset:
            predicted_answer = model.predict(question, gold_paragraphs)['answer']

            em = exact_match(gold_answer, predicted_answer)
            f1 = f1_score(gold_answer, predicted_answer)

            em_metric.append(em)
            f1_metric.append(f1)

            em_avg = sum(em_metric) / len(em_metric)
            f1_avg = sum(f1_metric) / len(f1_metric)

            progress_bar.set_description(desc=f'EM: {em:.2f}  F1: {f1:.2f}  Avg EM: {em_avg:.2f}  Avg F1: {f1_avg:.2f}')
            progress_bar.update()

    em_final = sum(em_metric) / len(em_metric)
    f1_final = sum(f1_metric) / len(f1_metric)
    print(f'Final metrics\tEM: {em_final:.2f}\t F1: {f1_final:.2f}')


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
