import argparse
import contextlib
import json
import os
from functools import partial
from multiprocessing import Manager, Pool

from tqdm.auto import tqdm

from graphqa.core import (
    ElasticGraphRetriever,
    RecurrentReasoner,
    TfIdfGraphRetriever,
    TransformerReader,
    Pipeline,
)
from graphqa.core.metrics import exact_match, f1_score


pipeline = None


def get_retriever(model_name, model_path, *args, **kwargs):
    models = {
        'gensim': TfIdfGraphRetriever,
        'elastic': ElasticGraphRetriever,
    }
    model = models[model_name](*args, **kwargs)
    model.load(model_path)
    return model


def get_reader(model_name, model_path, *args, **kwargs):
    models = {
        'transformer': TransformerReader,
    }
    model = models[model_name](*args, **kwargs)
    model.load(model_path)
    return model


def get_reasoner(model_name, model_path, *args, **kwargs):
    models = {
        'recurrent': RecurrentReasoner,
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

    eval_dataset = []
    for entry in dataset:
        question = entry['question']
        answer = entry['answer']
        eval_dataset.append((question, answer))
    return eval_dataset


def create_pipeline(retriever_args, reasoner_args, reader_args):
    retriever = get_retriever(**retriever_args)
    reasoner = get_reasoner(**reasoner_args)
    reader = get_reader(**reader_args)
    return Pipeline(retriever, reasoner, reader)


def init_pipeline(**kwargs):
    component_args = {
        'retriever_args': {},
        'reasoner_args': {},
        'reader_args': {},
    }

    for key, value in kwargs.items():
        tokens = key.split('_')
        component = tokens[0]
        arg = '_'.join(tokens[1:])
        component_args[f'{component}_args'][arg] = value

    return create_pipeline(**component_args)


def main(dataset_name, dataset_path, num_workers=1, **kwargs):
    dataset = load_dataset(dataset_name, dataset_path)

    em_metric, f1_metric = [], []
    pipeline = init_pipeline(**kwargs)

    with tqdm(desc='Evaluating', unit='p', total=len(dataset), dynamic_ncols=True) as progress_bar:
        for question, gold_answer in dataset:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                predicted_answer = pipeline(question)['answer']

            em = exact_match(gold_answer, predicted_answer)
            f1 = f1_score(gold_answer, predicted_answer)

            em_metric.append(em)
            f1_metric.append(f1)

            em_avg = sum(em_metric) / len(em_metric)
            f1_avg = sum(f1_metric) / len(f1_metric)

            progress_bar.set_description(desc=f'EM: {em:.2f}  F1: {f1:.2f}  Avg EM: {em_avg:.2f}  Avg F1: {f1_avg:.2f}')
            # progress_bar.write(f'Gold: {gold_answer}\nPredicted: {predicted_answer}')
            progress_bar.update()

    em_final = sum(em_metric) / len(em_metric)
    f1_final = sum(f1_metric) / len(f1_metric)
    print(f'Final metrics\tEM: {em_final:.2f}\t F1: {f1_final:.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='System evaluation script.')

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
        '--retriever-model-name',
        metavar='Retriever model name',
        type=str,
        nargs='?',
        help="Retriever model name.",
        default='elastic',
        choices=['tfidf', 'elastic'],
    )

    parser.add_argument(
        '--retriever-model-path',
        metavar='Retriever model  path',
        type=str,
        nargs='?',
        help="Path from which to load retriever model.",
        default=None,
    )

    parser.add_argument(
        '--retriever-num-best',
        metavar='Number of matching paragraphs',
        type=int,
        nargs='?',
        help="Number of initial paragraphs to retrieve",
        default=10,
    )

    parser.add_argument(
        '--retriever-num-related',
        metavar='Number of related paragraphs',
        type=int,
        nargs='?',
        help="Number of connected paragraphs to retrieve for multihop.",
        default=50,
    )

    parser.add_argument(
        '--retriever-max-path-depth',
        metavar='Max path depth',
        type=int,
        nargs='?',
        help="Maximal distance between nodes in Graph DB.",
        default=2,
    )

    parser.add_argument(
        '--retriever-elastic-address',
        metavar='Elasticsearch address',
        type=str,
        nargs='?',
        help="Elasticsearch address",
        default='localhost:9200',
        dest='retriever_es_addr'
    )

    parser.add_argument(
        '--retriever-graph-db-address',
        metavar='Graph DB address',
        type=str,
        nargs='?',
        help="Graph DB address",
        default='bolt://localhost:7687',
        dest='retriever_db_addr'
    )

    parser.add_argument(
        '--retriever-index-name',
        metavar='Index name',
        type=str,
        nargs='?',
        help="Name of the index in Elasticsearch DB.",
        default='paragraphs',
    )

    parser.add_argument(
        '--reasoner-model-name',
        metavar='Reasoner model name',
        type=str,
        nargs='?',
        help="Reasoner model name.",
        default='recurrent',
        choices=['recurrent'],
    )

    parser.add_argument(
        '--reasoner-model-path',
        metavar='Reasoner model  path',
        type=str,
        nargs='?',
        help="Path from which to load reasoner model.",
        default=None,
    )

    parser.add_argument(
        '--reasoner-pretrained-weights',
        metavar='Reasoner pretrained weights',
        type=str,
        nargs='?',
        help="Pretrained weights for reasoner encoder.",
        default='bert-base-uncased',
        dest='reasoner_pretrained'
    )

    parser.add_argument(
        '--reasoner-num-reasoning-steps',
        metavar='Number of reasoning steps',
        type=int,
        nargs='?',
        help="Maximal number of reasoning steps to consider.",
        default=2,
    )

    parser.add_argument(
        '--reasoner-max-paragraph-num',
        metavar='Number of paragraphs',
        type=int,
        nargs='?',
        help="Number of paragraphs to use for reasoning.",
        default=10,
    )

    parser.add_argument(
        '--reasoner-max-seq-len',
        metavar='Max sequence length',
        type=int,
        nargs='?',
        help="Maximal paragraph length.",
        default=192,
    )

    parser.add_argument(
        '--reader-model-name',
        metavar='Reader model name',
        type=str,
        nargs='?',
        help="Reader model name.",
        default='transformer',
        choices=['transformer'],
    )

    parser.add_argument(
        '--reader-model-path',
        metavar='Reader model  path',
        type=str,
        nargs='?',
        help="Path from which to load reader model.",
        default=None,
    )

    args = vars(parser.parse_args())
    main(**args)
