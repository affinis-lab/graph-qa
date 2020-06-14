import argparse
import json

from tqdm.auto import tqdm

from graphqa.core import TransformerBinaryRanker


def load_model(model_name, model_path, *args, **kwargs):
    models = {
        'transformer': TransformerBinaryRanker,
    }
    model = models[model_name](*args, **kwargs)
    model.load(model_path)
    return model


def load_dataset(name, path):
    datasets = {
        'hotpotqa': load_hotpotqa,
    }
    return datasets[name](path)


def load_hotpotqa(path):
    with open(path) as f:
        return json.load(f)


def accuracy_and_recall_at_n(model, dataset, n=4):
    accuracy, recall_at_n = [], []
    for entry in tqdm(dataset, desc='Evaluating'):
        question = entry['question']
        paragraphs = list(map(lambda x: (x[0], ''.join(x[1])), entry['context']))
        gold_titles = set(map(lambda x: x[0], entry['supporting_facts']))

        # apply paragraph selection model on each paragraph
        accepted, rejected = [], []
        for paragraph_title, paragraph_text in paragraphs:
            prediction, confidence = model.score(question, paragraph_text)
            if prediction == 1: # collect selected paragraphs
                accepted.append((paragraph_title, confidence))
            else:
                rejected.append((paragraph_title, confidence))

        # sort by confidence descending
        accepted = sorted(accepted, key=lambda x: x[1], reverse=True)

        # we only care about top n results
        accepted = accepted[:n]

        # count how many gold paragraphs appear in top n
        counter = 0
        for result in accepted:
            if result[0] in gold_titles:
                counter += 1
                accuracy.append(1)
            else:
                accuracy.append(0)

        for result in rejected:
            accuracy.append(0 if result[0] in gold_titles else 1)

        # collect the fraction of gold paragraphs in top n
        recall_at_n.append(counter / len(gold_titles))

    # return mean metrics
    return {
        'acc': sum(accuracy) / len(accuracy),
        'recall@n': sum(recall_at_n) / len(recall_at_n),
    }


def main(dataset_name, dataset_path, model_name, model_path, n=4, *args, **kwargs):
    dataset = load_dataset(dataset_name, dataset_path)
    model = load_model(model_name, model_path, *args, **kwargs)

    metrics = accuracy_and_recall_at_n(model, dataset, n)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ranker evaluation script.')

    parser.add_argument(
        'dataset_name',
        metavar='Dataset name',
        type=str,
        help="Dataset name determines what loader to use.",
        choices=['hotpotqa'],
    )

    parser.add_argument(
        'dataset_path',
        metavar='Dataset path',
        type=str,
        help="Path from which to load dataset.",
    )

    parser.add_argument(
        'model_name',
        metavar='Model path',
        type=str,
        help="Name of the model to evaluate.",
        choices=['transformer']
    )

    parser.add_argument(
        '--model-path',
        metavar='Model path',
        type=str,
        nargs='?',
        help="Path from which to load model.",
        default=None,
    )

    parser.add_argument(
        '-n',
        metavar='Top N',
        type=int,
        nargs='?',
        help="Number of top paragraphs to consider.",
        default=4,
    )

    args = vars(parser.parse_args())
    main(**args)
