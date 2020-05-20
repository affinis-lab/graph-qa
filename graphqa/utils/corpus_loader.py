import bz2
import json
import re
import nltk

from constants import CORPUS_PATH


class CorpusLoader:
    def __init__(self):
        self.lanc_stemmer = nltk.stem.lancaster.LancasterStemmer()

    def load_corpus(self):        
        with bz2.open(CORPUS_PATH, "rb") as f:
            data = f.read()
        
        data = json.loads(self.make_json_valid(data))

        return self.create_corpus(data)

    def create_corpus(self, content):
        corpus = []
        for doc in content:
            # leave out title
            for passage in doc['text'][1:]:
                doc_string = ''
                for sent in passage:
                    doc_string += self.clean_a_tags(sent)
                corpus.append(str(doc_string.encode('utf-8')))
        return corpus

    def make_json_valid(self, data):
        data = data.decode('utf-8')
        data = re.sub('\r?\n{', '\n,{', data)
        data = list(data)
        data[0:0] = ['[']
        data.append(']')
        return ''.join(data)

    def clean_a_tags(self, raw_text):
        regex = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleaned_text = re.sub(regex, '', raw_text)
        return cleaned_text

    def preprocess(self, s):
        return ' '.join(list(self.lanc_stemmer.stem(w) for w in nltk.word_tokenize(s)))