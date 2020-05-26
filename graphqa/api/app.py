import os
from functools import partial

import nltk
from flask import (
    Flask,
    jsonify,
    request,
)
from nltk.corpus import stopwords

from graphqa.api.constants import (
    DEFAULT_LANGUAGE,
    GRAPH_DB,
    GRAPH_QA_API_CONFIG_PATH,
    READER_ARCHITECTURE,
    READER_PATH,
    RETRIEVER_PATH,
)

from graphqa.core import (
    TransformerReader,
    DummyReasoner,
    Pipeline,
    TfIdfGraphRetriever,
)

from graphqa.core.utils import tokenize


app = Flask(__name__)

# load default config
app.config.from_pyfile('config.py')
# load deployment config if provided
if os.environ.get(GRAPH_QA_API_CONFIG_PATH):
    app.config.from_envvar(GRAPH_QA_API_CONFIG_PATH)


@app.before_first_request
def init():
    tokenizer = partial(tokenize, stopwords=set(stopwords.words(DEFAULT_LANGUAGE)))

    db_addr = app.config[GRAPH_DB]
    retriever_path = app.config[RETRIEVER_PATH]
    retriever = TfIdfGraphRetriever(db=db_addr, tokenizer=tokenizer)
    retriever.load(retriever_path)

    reasoner = DummyReasoner()

    reader_path = app.config[READER_PATH]

    reader = TransformerReader()
    reader.load(reader_path)

    app.pipeline = Pipeline(retriever, reasoner, reader)


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data['question']
    result = app.pipeline(question)
    return jsonify({
        'answer': result['answer'],
        'confidence': result['confidence'],
        'context': result['context'],
        'supportingFacts': result['supporting_facts'],
    })
