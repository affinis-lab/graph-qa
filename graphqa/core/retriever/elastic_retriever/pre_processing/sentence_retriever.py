import os
import json
import logging
import pandas as pd

from argparse import ArgumentParser
from typing import Dict
from spacy.lang.en import English

logging.basicConfig(level=logging.INFO)


def save_topic(topic_dictionary: Dict[str, Dict],
               topic_id: str,
               save_folder: str) -> None:
    # this check is for the very first topic id
    if topic_id == "":
        return
    topic_to_be_saved = topic_dictionary[topic_id]
    file_name = topic_id + ".json"
    full_save_path = os.path.join(save_folder, file_name)

    with open(full_save_path, 'w') as fp:
        json.dump(topic_to_be_saved, fp, indent=4)

    logging.info("finished with id: ", topic_id)


def parse_wiki_paragraphs(text_path, save_path):
    data = pd.read_csv(text_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    topic_dictionary = dict()
    current_id = ""
    counter = 0
    for index, row in data.iterrows():
        counter += 1

        # for testing purposes
        # if counter > 1000:
        #     break
        row_id = row[":ID"]
        if "-0" in row_id:
            save_topic(topic_dictionary, current_id, save_path)

            logging.info("Current row id: ", row_id)
            current_id = row_id
            topic_name = row["text:string"]
            topic_dictionary[row_id] = dict()
            topic_dictionary[row_id]["topic_name"] = topic_name
            topic_dictionary[row_id]["paragraphs"] = dict()
        else:
            topic_text = row["text:string"]
            paragraph = dict()
            paragraph["text"] = topic_text
            try:
                paragraph["sentences"] = get_sentences(topic_text)
            except:
                # empty paragraph
                continue
            topic_dictionary[current_id]["paragraphs"][row_id] = paragraph
    # with open(save_path, 'w') as fp:
    #     json.dump(topic_dictionary, fp, indent=4)


def get_sentences(text):
    return_list = []
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    parsed_text = nlp(text)
    for sentence in parsed_text.sents:
        return_list.append(sentence.text)
    return return_list


def main(arguments):
    csv_path = arguments.csv
    save_dir = arguments.save
    parse_wiki_paragraphs(csv_path,
                          save_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv', required=True, help='path to CSV input file (paragraph.csv)')
    parser.add_argument('--save', required=True, help='path where to save processed JSON files')
    args = parser.parse_args()

    main(args)
