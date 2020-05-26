import nltk


def tokenize(text, stopwords=[]):
    tokens = nltk.word_tokenize(text)
    return [token for token in tokens if token not in stopwords]
