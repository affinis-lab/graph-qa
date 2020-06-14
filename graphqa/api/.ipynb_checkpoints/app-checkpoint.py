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
    HGN_PATH,
    RANKER_PATH,
    RETRIEVER_ELASTICSEARCH_ADDR,
    RETRIEVER_ELASTICSEARCH_INDEX,
    RETRIEVER_GRAPH_DB,
    RETRIEVER_PATH,
)

from graphqa.core import (
    TransformerBinaryRanker,
    RecurrentReasoner,
    HGNPipeline,
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

    ranker_path = app.config[RANKER_PATH]
    ranker = TransformerBinaryRanker()
    ranker.load(ranker_path)
    
    hgn_path = app.config[HGN_PATH]
    hgn = HGN('bert-base-uncased', model_path=hgn_path)

    app.pipeline = HGNPipeline(retriever, ranker, hgn)


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
