from modules.pipeline import Pipeline
from modules.reader.bert_reader import BertReader
from modules.reasoner.binary_classifier_reasoner import BinaryClassifierReasoner
from modules.retriever.tf_idf_retriever import TfIdfRetriever

# TODO trenutno se ucitava samo jedan .bz2 fajl, napraviti da se ucitavaju svi (onoliko koliko moze nas RAM da podnese) .bz2 fajlovi u svim folderima 
# TODO 

def main():

    corpus_path = 'enwiki-20171001-pages-meta-current-withlinks-processed/AE/wiki_00.bz2'
    reasoner_model_path = 'models/albert-binary-classifier'
    reader_model_path = 'models/squad-qa-model'
    retriever = TfIdfRetriever(corpus_path, '')
    reasoner = BinaryClassifierReasoner(reasoner_model_path)
    reader = BertReader(reader_model_path)

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

    _, kwargs = pipeline(question="""What was Henry VIII's ship?""", paragraphs=[])

    print(kwargs['answer'])


if __name__ == '__main__':
    main()