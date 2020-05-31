import csv
import os
import re
import time
import sys
from functools import partial

import pandas as pd
import nltk
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import Similarity
from nltk.corpus import stopwords


def load_data(path):
    with open(path) as f:
        next(f)
        yield from csv.reader(f)

def split_id_paragraph(data_iterator):
    for line in data_iterator:
        yield line[0], line[3]

def get_ids(data_iterator):
    for idx, (key, _) in enumerate(data_iterator):
        print(f'\rProcessing paragraph {idx}', end='')
        yield key

def get_text(data_iterator):
    for idx, (_, text) in enumerate(data_iterator):
        print(f'\rProcessing paragraph {idx}', end='')
        yield text

def load_ids(path):
    return get_ids(split_id_paragraph(load_data(path)))

def tokenize(text, stopwords=[]):
    tokens = nltk.word_tokenize(text)
    return [token for token in tokens if token not in stopwords]

def clean_paragraph(paragraph):
    paragraph = re.sub('\W+', ' ', paragraph)
    paragraph = re.sub('\s+', ' ', paragraph)
    return paragraph.strip()

def is_ok(paragraph):
    if not type(paragraph) == str:
        return False
    return bool(paragraph)

def load_paragraphs(path):
    tokenizer = partial(tokenize, stopwords=set(stopwords.words('english')))
    paragraphs = get_text(split_id_paragraph(load_data(path)))
    paragraphs = filter(is_ok, paragraphs)
    paragraphs = map(clean_paragraph, paragraphs)
    paragraphs = map(tokenizer, paragraphs)
    return paragraphs

def main(dataset_path):
    if not os.path.exists('../data/retriever/paragraph-ids.txt'):
        print('Writing paragraph ID to file...')
        with open('../data/retriever/paragraph-ids.txt', 'w') as f:
            for paragraph_id in load_ids(dataset_path):
                f.write(paragraph_id + '\n')

    dictionary_path = '../data/retriever/dct.pkl'
    if not os.path.exists(dictionary_path):
        print('Creating dictionary...')
        st = time.time()
        dct = Dictionary(load_paragraphs(dataset_path), prune_at=5000000)
        dct.save(dictionary_path, pickle_protocol=3)
        et = time.time()
        print(f'\rFinished creating dictionary in {et - st}s.')
    else:
        print('Loading dictionary...')
        dct = Dictionary.load(dictionary_path)
        print('Dictionary loaded.')

    tfidf_path = '../data/retriever/tfidf.pkl'
    if not os.path.exists(tfidf_path):
        print('Creating model...')
        st = time.time()
        corpus = map(dct.doc2bow, load_paragraphs(dataset_path))
        model = TfidfModel(corpus)
        model.save(tfidf_path, pickle_protocol=3)
        et = time.time()
        print(f'\rFinished creating model in {et - st}s.')
    else:
        print('Loading model...')
        model = TfidfModel.load(tfidf_path)
        print('Model loaded.')

    index_path = '../data/retriever/indexes/master-index'
    if not os.path.exists(index_path):
        print('Creating index...')
        st = time.time()
        corpus = map(dct.doc2bow, load_paragraphs(dataset_path))
        index = Similarity('../data/retriever/indexes/index', model[corpus], len(dct))
        index.save(index_path)
        et = time.time()
        print(f'\rFinished creating index in {et - st}s.')
        print('Done')
    else:
        print('Nothing to do. Exiting...')


if __name__ == "__main__":
    main(sys.argv[1])
