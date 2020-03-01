from modules.pipeline import Pipeline
from modules.reader.bert_reader import BertReader
from modules.reasoner.binary_classifier_reasoner import BinaryClassifierReasoner
from modules.retriever.tf_idf_retriever import TfIdfRetriever

from utils.redis_db import RedisDB

# TODO trenutno se ucitava samo jedan .bz2 fajl, napraviti da se ucitavaju svi (onoliko koliko moze nas RAM da podnese) .bz2 fajlovi u svim folderima 
# TODO 

def main():
    retriever = TfIdfRetriever()
    reasoner = BinaryClassifierReasoner()
    reader = BertReader()

    pipeline_components = [
        retriever, reasoner, reader
    ]

    pipeline = Pipeline(pipeline_components)
    # When did Ferguson join Manchester United? - T
    # Who did Manchester United beat in the 2008 Champions League finals?
    # Who did Manchester United beat to win the 2008 European Cup?
    # Who is the first manager to win Premier League three consecutive times? - T
    # What is the Bardet-Biedl syndrome? - T
    # What is the name of the largest ship in Norway's military? - T
    # What is the name of the largest ship in Norway's army? - T
    # For how many years did Galileo orbit Jupiter? - kanal
    # After who is the Europa moon named?
    # What were Oberon and Titania named after? - recimo T
    # What is the name of the Saturn's largest moon? - T
    # Who produced first colored photographs? - T
    # What was the name of Henry VIII's ship? - F
    # What was Henry VIII's ship? - T

    _, kwargs = pipeline(question="""Who produced first colored photographs?""", paragraphs=[])

    print(kwargs['answer'])


if __name__ == '__main__':
    main()