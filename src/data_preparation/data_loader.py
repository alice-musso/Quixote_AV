import os
import pickle
from pathlib import Path
from typing import List
import spacy
import re
from collections import Counter
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


# ------------------------------------------------------------------------
# document loading routine
# ------------------------------------------------------------------------
import nltk
from nltk.corpus import stopwords

from src.data_preparation.segmentation import Segmentator




def get_spanish_function_words():
    stop_words_sp = set(stopwords.words('spanish'))
    return stop_words_sp


def get_author_from_path(path):
    return path.name.split('-')[0].strip()


def get_bookname_from_path(path):
    return path.name.split('-')[1].strip()


# ----------------------------------------------
# Data helpers
# ----------------------------------------------
class Book:

    def __init__(self, path):
        author, title = path.stem.split('-')
        raw_text = path.read_text(encoding='utf8', errors='ignore')
        clean_text = self._clean_text(raw_text)

        self.path = path
        self.title = title.strip()
        self.author = author.strip()
        self.raw_text = raw_text
        self.clean_text = clean_text
        self.processed = None
        self.fragments = None

    def _clean_text(self, text):
        """Clean and normalize text content."""
        print('REMINDER: check clean text')
        # text = text.lower()
        text = re.sub(r'\{[^{}]*\}', '', text)
        text = re.sub(r'\*[^**]*\*', '', text)
        text = re.sub(r'<\w>(.*?)</\w>', r'\1', text)
        text = text.replace('\x00', '')
        return text.strip()


    def __repr__(self):
        return f'({self.author}) "{self.title}"'


class DocumentProcessor:

    def __init__(self, language_model="es_dep_news_trf", language_model_length = 1_200_000, savecache='./data_preparation/.cache/processed_docs.pkl'):
        self.language_model = language_model
        self.language_model_length = language_model_length
        self.nlp = None  # lazy load
        self.savecache = savecache
        self.init_cache()

    def get_nlp(self):
        if self.nlp is None:
            print('loading spacy model...')
            self.nlp = spacy.load(self.language_model)
            self.nlp.max_length = self.language_model_length
            print('[spacy loaded]')
        return self.nlp

    def init_cache(self):
        if self.savecache is None or not os.path.exists(self.savecache):
            print('Cache not found, initializing from scratch')
            self.cache = {}
        else:
            print(f'Loading cache from {self.savecache}')
            self.cache = pickle.load(open(self.savecache, 'rb'))

    def save_cache(self):
        if self.savecache is not None:
            print(f'Storing cache in {self.savecache}')
            parent = Path(self.savecache).parent
            if parent:
                os.makedirs(parent, exist_ok=True)
            pickle.dump(self.cache, open(self.savecache, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def process_document(self, document, cache_idx):
        if cache_idx not in self.cache:
            print(f'{cache_idx} not in cache')
            processed_doc = self.get_nlp()(document)
            self.cache[cache_idx] = processed_doc
            self.save_cache()
        processed_doc = self.cache[cache_idx]
        return processed_doc


def _job_open_book(file):

    cache_idx = Path(file).name.replace(' ','_')
    processor = DocumentProcessor(savecache=f'./data_preparation/.cache/processed_doc_{cache_idx}.pkl')
    segmentator = Segmentator()

    book = Book(file)

    # spacy processing of the full document
    book.processed = processor.process_document(book.clean_text, cache_idx)

    # segmentation
    book.segmented = segmentator.transform(book.processed)

    return book


def load_corpus(path: str):

    multiprocessing.set_start_method("spawn", force=True)

    paths = Path(path).glob('*.txt')
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(_job_open_book, p): p for p in paths}
        corpus = []
        for future in as_completed(futures):
            corpus.append(future.result())

    authors = set([book.author for book in corpus])

    print(f'Total documents: {len(corpus)}')
    print(f'Total authors: {len(authors)}')

    return corpus


def binarize_corpus(corpus: List[Book], positive_author='Cervantes'):
    for book in corpus:
        if book.author != positive_author:
            book.author = 'Not' + positive_author
    return corpus


def remove_unique_authors(corpus: List[Book]):
    counts = Counter(book.author for book in corpus)
    return [book for book in corpus if counts[book.author]>1]


def _remove_single_author_texts(corpus: list[dict]) -> list[dict]:
    """Remove texts by authors who only have one work."""
    author_counts = Counter(doc['author'] for doc in corpus)
    return [doc for doc in corpus if author_counts[doc['author']] > 1]








