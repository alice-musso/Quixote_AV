import os
import sys
from os.path import join
import collections
from glob import glob, escape
from pathlib import Path
from itertools import chain
import numpy as np
from tqdm import tqdm
import re
from collections import Counter
from sympy import false

# ------------------------------------------------------------------------
# document loading routine
# ------------------------------------------------------------------------
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def get_spanish_function_words():
    stop_words_sp = set(stopwords.words('spanish'))
    return stop_words_sp

# data_loader.py

def load_corpus(path: str, remove_unique_authors, remove_quixote, remove_avellaneda, test_documents) -> tuple[list[str], list[str], list[str]]:
    """Load corpus documents with optional filtering.
    
    Args:
        path: Directory path containing corpus files
        TODO repair
        - remove_test: Remove test document (Quaestio)
        - remove_unique_authors: Remove texts by authors with single work

    Returns:
        Tuple of (documents, authors, filenames)
    """

    files = [f for f in Path(path).glob('*.txt')]
    corpus = []

    def get_author_from_path(path):
        return path.name.split('-')[0].strip()

    def get_bookname_from_path(path):
        return path.name.split('-')[1].strip()

    if remove_avellaneda:
        files = [f for f in files if get_author_from_path(f) != 'Avellaneda']

    if remove_quixote:
        files = [f for f in files if 'Quijote' not in get_bookname_from_path(f)]

    if remove_unique_authors:
        counts = Counter(get_author_from_path(f) for f in files)
        to_keep = [f for f in files if counts[get_author_from_path(f)]>1]
        files = to_keep

    # adds the test documents, no matter if removed previously
    string_document_set = set(f.name for f in files)
    if isinstance(test_documents, str):
        test_documents = [test_documents]
    for test in test_documents:
        if test not in string_document_set:
            files += [Path(join(path, test +".txt"))]


    for file in tqdm(files, desc=f'Loading corpus from {path}'):

        author, title = file.stem.split('-')
        text = _clean_text(file.read_text(encoding='utf8', errors='ignore'))
        
        corpus.append({
            'text': text,
            'author': author.strip(),
            'filename': file.stem
        })

    # if filters.get('remove_unique_authors'):
    #     corpus = _remove_single_author_texts(corpus)

    documents = [doc['text'] for doc in corpus]
    authors = [doc['author'] for doc in corpus]
    filenames = [doc['filename'] for doc in corpus]

    print(f'Total documents: {len(documents)}')
    print(f'Total authors: {len(set(authors))}')
    
    return documents, authors, filenames

# def _should_skip_file(filename: str, filters: dict) -> bool:
#     """Check if file should be filtered out based on criteria."""
#     checks = {
#         'remove_epistles': lambda f: 'epistola' in f.lower(),
#         'remove_egloghe': lambda f: 'egloga' in f.lower(),
#         'remove_anonymus_files': lambda f: any(x in f.lower() for x in ['misc', 'anonymus']),
#         'remove_monarchia': lambda f: 'monarchia' in f.lower(),
#         'remove_quijote': lambda f: ('cervantes' in f.lower() and 'don quijote' in f.lower()),
#         'remove_test': lambda f: 'avellaneda' in f.lower(),
#     }
#     active_flags = [flag for flag in filters if filters.get(flag)]
#     print(f'Checking file: {filename}, active filters: {active_flags}')
#
#     return any(check(filename) for flag, check in checks.items() if filters.get(flag))

def _clean_text(text: str) -> str:
    """Clean and normalize text content."""
    text = text.lower()
    text = re.sub(r'\{[^{}]*\}', '', text)
    text = re.sub(r'\*[^**]*\*', '', text) 
    text = re.sub(r'<\w>(.*?)</\w>', r'\1', text)
    text = text.replace('\x00', '')
    return text.strip()

def _remove_single_author_texts(corpus: list[dict]) -> list[dict]:
    """Remove texts by authors who only have one work."""
    author_counts = Counter(doc['author'] for doc in corpus)
    return [doc for doc in corpus if author_counts[doc['author']] > 1]








