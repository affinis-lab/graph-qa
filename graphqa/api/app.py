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
    GRAPH_QA_API_CONFIG_PATH,
    READER_ARCHITECTURE,
    READER_PATH,
    RETRIEVER_ELASTICSEARCH_ADDR,
    RETRIEVER_ELASTICSEARCH_INDEX,
    RETRIEVER_GRAPH_DB,
    RETRIEVER_PATH,
    REASONER_ARCHITECTURE,
    REASONER_PATH,
    REASONER_NUM_REASONING_STEPS,
    REASONER_MAX_PARAGRAPH_NUM,
    REASONER_MAX_SEQ_LEN
)

from graphqa.core import (
    TransformerReader,
    RecurrentReasoner,
    Pipeline,
    ElasticGraphRetriever,
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
    db_addr = app.config[RETRIEVER_GRAPH_DB]
    es_addr = app.config[RETRIEVER_ELASTICSEARCH_ADDR]
    index_name = app.config[RETRIEVER_ELASTICSEARCH_INDEX]
    retriever = ElasticGraphRetriever(
        db_addr, es_addr, index_name, num_best=10, num_related=50
    )

    reasoner_architecture = app.config[REASONER_ARCHITECTURE]
    reasoner_path = app.config[REASONER_PATH]
    reasoner_num_reasoning_steps = app.config[REASONER_NUM_REASONING_STEPS]
    reasoner_max_paragraph_num = app.config[REASONER_MAX_PARAGRAPH_NUM]
    reasoner_max_seq_len = app.config[REASONER_MAX_SEQ_LEN]

    reasoner = RecurrentReasoner(
        reasoner_architecture,
        model_path=reasoner_path,
        num_reasoning_steps=reasoner_num_reasoning_steps,
        max_paragraph_num=reasoner_max_paragraph_num,
        max_seq_len=reasoner_max_seq_len,
    )

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
