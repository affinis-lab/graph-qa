import warnings
warnings.simplefilter('ignore')

import bz2
import csv
import glob
import json
import urllib.parse
import uuid
import subprocess
import sys
from functools import partial
from itertools import chain
from multiprocessing import Manager, Pool
from pathlib import Path

from bs4 import BeautifulSoup
from tqdm.auto import tqdm


PARAGRAPH_NODE = 'paragraph'
SENTENCE_NODE = 'sentence'
ENTITY_NODE = 'entity'

NODE_TYPES = [PARAGRAPH_NODE, SENTENCE_NODE]
NODE_HEADERS = {
    PARAGRAPH_NODE: 'paragraphId:ID,documentTitle:string,documentUrl:string,text:string,:LABEL',
    SENTENCE_NODE: 'sentenceId:ID,text:string,:LABEL',
}

PARAGRAPH_PRECEDED_BY_EDGE = 'paragraph_preceded_by'
PARAGRAPH_FOLLOWED_BY_EDGE = 'paragraph_followed_by'
PARAGRAPH_SUMMARIZED_BY_EDGE = 'paragraph_summarized_by'
PARAGRAPH_LINKS_TO_EDGE = 'paragraph_links_to'

SENTENCE_OF_EDGE = 'sentence_of'
SENTENCE_PRECEDED_BY_EDGE = 'sentence_preceded_by'
SENTENCE_FOLLOWED_BY_EDGE = 'sentence_followed_by'
SENTENCE_LINKS_TO_EDGE = 'sentence_links_to'

ENTITY_OF_EDGE = 'entity_of'
ENTITY_LINKS_TO = 'entity_links_to'

DEFAULT_EDGE_HEADER = ':START_ID,:END_ID,:TYPE'

EDGE_TYPES = [
    PARAGRAPH_PRECEDED_BY_EDGE,
    PARAGRAPH_FOLLOWED_BY_EDGE,
    PARAGRAPH_SUMMARIZED_BY_EDGE,
    PARAGRAPH_LINKS_TO_EDGE,
    SENTENCE_OF_EDGE,
    SENTENCE_PRECEDED_BY_EDGE,
    SENTENCE_FOLLOWED_BY_EDGE,
    SENTENCE_LINKS_TO_EDGE,
    ENTITY_OF_EDGE,
    ENTITY_LINKS_TO,
]
EDGE_HEADERS = {
    PARAGRAPH_PRECEDED_BY_EDGE: DEFAULT_EDGE_HEADER,
    PARAGRAPH_FOLLOWED_BY_EDGE: DEFAULT_EDGE_HEADER,
    PARAGRAPH_SUMMARIZED_BY_EDGE: DEFAULT_EDGE_HEADER,
    PARAGRAPH_LINKS_TO_EDGE: DEFAULT_EDGE_HEADER,
    SENTENCE_OF_EDGE: DEFAULT_EDGE_HEADER,
    SENTENCE_PRECEDED_BY_EDGE: DEFAULT_EDGE_HEADER,
    SENTENCE_FOLLOWED_BY_EDGE: DEFAULT_EDGE_HEADER,
    SENTENCE_LINKS_TO_EDGE: DEFAULT_EDGE_HEADER,
    ENTITY_OF_EDGE: DEFAULT_EDGE_HEADER,
    ENTITY_LINKS_TO: DEFAULT_EDGE_HEADER,
}


def process_wiki(input_dir, output_dir, chunksize=64):
    dataset_files = glob.glob(f'{input_dir}/**/*.bz2')
    print(f'Found {len(dataset_files)} bz2 files to process.')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    node_output_path = output_path / 'nodes'
    edge_output_path = output_path / 'edges'

    with Manager() as manager:
        document_hrefs = manager.dict()
        with Pool() as p:
            paths = list(map(lambda path: (path, document_hrefs), dataset_files))
            desc = 'Collecting document hrefs'
            with tqdm(desc=desc, total=len(paths), unit='file') as progress_bar:
                for _ in p.imap(collect_document_hrefs, paths, chunksize=chunksize):
                    progress_bar.update()

        with Pool() as p:
            extract_node_args = (document_hrefs, node_output_path, edge_output_path)
            paths = list(map(lambda args: args + extract_node_args, enumerate(dataset_files)))
            desc = 'Creating nodes and edges'
            with tqdm(desc=desc, total=len(paths), unit='file') as progress_bar:
                for _ in p.imap(extract_nodes_and_edges, paths, chunksize=chunksize):
                    progress_bar.update()

    def get_args(output_dir, headers, typ):
        return (output_dir / typ, output_path / f'{typ}.csv', headers[typ])

    with Pool() as p:
        format_node_path = lambda typ: get_args(node_output_path, NODE_HEADERS, typ)
        node_files = list(map(format_node_path, NODE_TYPES))
        desc = 'Combining node files'
        with tqdm(desc=desc, total=len(node_files), unit='file') as progress_bar:
            for _ in p.imap(combine_files, node_files):
                progress_bar.update()

        format_edge_path = lambda typ: get_args(edge_output_path, EDGE_HEADERS, typ)
        edge_files = list(map(format_edge_path, EDGE_TYPES))
        desc = 'Combining edge files'
        with tqdm(desc=desc, total=len(edge_files), unit='file') as progress_bar:
            for _ in p.imap(combine_files, edge_files):
                progress_bar.update()


def collect_document_hrefs(args):
    dataset_path, hrefs = args

    with bz2.open(dataset_path, 'r') as f:
        dataset = f.readlines()

        documents = list(map(extract_document, dataset))
        for document in documents:
            hrefs[document["href"]] = document["id"]


def extract_nodes_and_edges(args):
    job_index, dataset_path, document_hrefs, node_output_path, edge_output_path = args

    with bz2.open(dataset_path, 'r') as f:
        dataset = f.readlines()

    file_number = str(job_index).zfill(5)

    def format_file_name(file_name, output_path):
        return f'{output_path}/{file_name}/{file_number}.csv'

    format_node_file_name = partial(format_file_name, output_path=node_output_path)
    format_edge_file_name = partial(format_file_name, output_path=edge_output_path)

    documents = list(map(extract_document, dataset))

    extract_paragraphs_func = partial(extract_paragraphs, document_hrefs=document_hrefs)
    paragraphs = list(chain.from_iterable(map(extract_paragraphs_func, documents)))

    paragraph_preceded_by_edges = map(get_paragraph_preceded_by_edges, paragraphs)
    save_csv(paragraph_preceded_by_edges, format_edge_file_name(PARAGRAPH_PRECEDED_BY_EDGE))

    paragraph_followed_by_edges = map(get_paragraph_followed_by_edges, paragraphs)
    save_csv(paragraph_followed_by_edges, format_edge_file_name(PARAGRAPH_FOLLOWED_BY_EDGE))

    paragraph_summarized_by_edges = map(get_paragraph_summarized_by_edges, paragraphs)
    save_csv(paragraph_summarized_by_edges, format_edge_file_name(PARAGRAPH_SUMMARIZED_BY_EDGE))

    paragraph_nodes = map(prepare_paragraph_node, paragraphs)
    save_csv(paragraph_nodes, format_node_file_name(PARAGRAPH_NODE))

    paragraph_links_to_edges = list(chain.from_iterable(map(get_paragraph_links_to_edge, paragraphs)))
    save_csv(paragraph_links_to_edges, format_edge_file_name(PARAGRAPH_LINKS_TO_EDGE))

    sentences = list(chain.from_iterable(map(lambda paragraph: paragraph['sentences'], paragraphs)))

    sentence_of_edges = map(get_sentence_of_edges, sentences)
    save_csv(sentence_of_edges, format_edge_file_name(SENTENCE_OF_EDGE))

    sentence_preceded_by_edges = map(get_sentence_preceded_by_edges, sentences)
    save_csv(sentence_preceded_by_edges, format_edge_file_name(SENTENCE_PRECEDED_BY_EDGE))

    sentence_followed_by_edges = map(get_sentence_followed_by_edges, sentences)
    save_csv(sentence_followed_by_edges, format_edge_file_name(SENTENCE_FOLLOWED_BY_EDGE))

    sentence_nodes = map(prepare_sentence_node, sentences)
    save_csv(sentence_nodes, format_node_file_name(SENTENCE_NODE))

    sentence_links_to_edges = list(chain.from_iterable(map(get_sentence_links_to_edge, sentences)))
    save_csv(sentence_links_to_edges, format_edge_file_name(SENTENCE_LINKS_TO_EDGE))


def combine_files(args):
    input_dir_path, output_file_path, output_headers = args
    input_file_paths = glob.glob(f'{input_dir_path}/*.csv')

    with open(output_file_path, 'w') as combined_file:
        combined_file.write(output_headers + '\n')

        for input_file_path in input_file_paths:
            with open(input_file_path) as input_file:
                combined_file.writelines(input_file.readlines())


def extract_document(data):
    document = json.loads(data)
    del document['charoffset']
    document['href'] = urllib.parse.quote(document['title']).lower()
    return document


def extract_paragraphs(document, document_hrefs):
    paragraphs = []
    paragraph_idx = 0
    has_intro = False

    text = document['text']
    text = text[1:] # skipping title
    for paragraph in text:
        paragraph_id = f'{document["id"]}-{paragraph_idx}'

        sentences = extract_sentences(paragraph, paragraph_id, document_hrefs)

        paragraph_text = ''.join(map(lambda sentence: sentence['text'], sentences))
        paragraph_links = set(chain.from_iterable(map(lambda sentence: sentence['links_to'], sentences)))

        if not paragraph_text:
            if has_intro:
                paragraph_idx += 1
            continue

        if not has_intro:
            has_intro = True
            intro_paragraph = paragraph_id

        if paragraphs:
            previous_paragraph = paragraphs[-1]
            previous_paragraph['next_paragraph'] = paragraph_id
            previous_paragraph_id = previous_paragraph['id']
        else:
            previous_paragraph_id = None

        paragraphs.append({
            'id': paragraph_id,
            'text': paragraph_text.strip(),
            'document_url': document['url'],
            'document_title': document['title'],
            'previous_paragraph': previous_paragraph_id,
            'next_paragraph': None,
            'intro_paragraph': intro_paragraph if intro_paragraph != paragraph_id else None,
            'sentences': sentences,
            'links_to': paragraph_links
        })

        paragraph_idx += 1

    return paragraphs


def extract_sentences(paragraph, paragraph_id, document_hrefs):
    sentences = []
    for sentence_idx, sentence_text in enumerate(paragraph):
        sentence_soup = BeautifulSoup(sentence_text, 'html.parser')
        sentence_raw = sentence_soup.text

        if not sentence_raw:
            continue

        sentence_id = f'{paragraph_id}-{sentence_idx}'

        if sentences:
            previous_sentence = sentences[-1]
            previous_sentence['next_sentence'] = sentence_id
            previous_sentence_id = previous_sentence['id']
        else:
            previous_sentence_id = None

        sentence_links = []
        for link_tag in sentence_soup.find_all('a', href=True):
            href = link_tag['href'].lower()
            if '#' in href:
                href = href[:href.find('#')]
            document_id = document_hrefs.get(href, None)
            if document_id:
                sentence_links.append(document_id)

        sentence = {
            'id': sentence_id,
            'text': sentence_raw,
            'previous_sentence': previous_sentence_id,
            'next_sentence': None,
            'links_to': sentence_links
        }

        sentences.append(sentence)

    return sentences


def prepare_paragraph_node(paragraph):
    return (
        paragraph['id'],
        paragraph['document_title'],
        paragraph['document_url'],
        paragraph['text'],
        PARAGRAPH_NODE.title()
    )


def prepare_sentence_node(sentence):
    return (
        sentence['id'],
        sentence['text'],
        SENTENCE_NODE.title()
    )


def get_document_href(document):
    return (document['id'], document['href'])


def get_paragraph_preceded_by_edges(paragraph):
    if not paragraph['previous_paragraph']:
        return None
    return (
        paragraph['id'],
        paragraph['previous_paragraph'],
        PARAGRAPH_PRECEDED_BY_EDGE.upper()
    )


def get_paragraph_followed_by_edges(paragraph):
    if not paragraph['next_paragraph']:
        return None
    return (
        paragraph['id'],
        paragraph['next_paragraph'],
        PARAGRAPH_FOLLOWED_BY_EDGE.upper()
    )


def get_paragraph_summarized_by_edges(paragraph):
    if not paragraph['intro_paragraph']:
        return None
    return (
        paragraph['id'],
        paragraph['intro_paragraph'],
        PARAGRAPH_SUMMARIZED_BY_EDGE.upper()
    )


def get_paragraph_links_to_edge(paragraph):
    links_to_edges = []
    for link in paragraph['links_to']:
        links_to_edges.append((paragraph["id"], f'{link}-0', PARAGRAPH_LINKS_TO_EDGE.upper()))
    return links_to_edges


def get_sentence_of_edges(sentence):
    paragraph_id = '-'.join(sentence['id'].split('-')[:-1])
    return (
        sentence['id'],
        paragraph_id,
        SENTENCE_OF_EDGE.upper()
    )


def get_sentence_preceded_by_edges(sentence):
    if not sentence['previous_sentence']:
        return None
    return(
        sentence['id'],
        sentence['previous_sentence'],
        SENTENCE_PRECEDED_BY_EDGE.upper()
    )


def get_sentence_followed_by_edges(sentence):
    if not sentence['next_sentence']:
        return None
    return (
        sentence['id'],
        sentence['next_sentence'],
        SENTENCE_FOLLOWED_BY_EDGE.upper()
    )


def get_sentence_links_to_edge(sentence):
    links_to_edges = []
    for link in sentence['links_to']:
        links_to_edges.append((sentence["id"], f'{link}-0', SENTENCE_LINKS_TO_EDGE.upper()))
    return links_to_edges


def save_csv(items, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        writer = csv.writer(
            f, doublequote=True, quoting=csv.QUOTE_ALL
        )
        for item in items:
            if not item:
                continue
            writer.writerow(item)


if __name__ == "__main__":
    process_wiki(sys.argv[1], sys.argv[2])
