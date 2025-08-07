import os
import pickle
from pathlib import Path
from typing import List
import spacy
from tqdm import tqdm
import re
from collections import Counter


# ------------------------------------------------------------------------
# document loading routine
# ------------------------------------------------------------------------
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

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

    def __init__(self, language_model=None, savecache='./data_preparation/.cache/processed_docs.pkl'):
        self.nlp = language_model
        self.savecache = savecache
        self.init_cache()

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

    def process_document(self, document, filename):
        if filename not in self.cache:
            print(f'{filename} not in cache')
            processed_doc = self.nlp(document)
            self.cache[filename] = processed_doc
            self.save_cache()
        processed_doc = self.cache[filename]
        return processed_doc

    # def process_documents(self, documents, filenames):
    #     processed_docs = {}
    #     for filename, doc in tqdm(zip(filenames, documents), total=len(filenames), desc='processing with spacy'):
    #         processed_docs[filename[:-2]] = self.process_document(doc, filename)
    #     return processed_docs


def load_corpus(path: str, spacy_language_model: 'SpaCy'):
    """Load corpus documents with optional filtering.

    Args:
        path: Directory path containing corpus files


    Returns:
        Tuple of (documents, authors, filenames)
    """

    processor = DocumentProcessor(language_model=spacy_language_model)

    corpus = []
    for file in tqdm(Path(path).glob('*.txt'), desc=f'Loading corpus from {path}'):
        book = Book(file)
        book.processed = processor.process_document(book.clean_text, Path(book.path).name)
        corpus.append(book)

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








