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

def load_corpus(path: str, **filters) -> tuple[list[str], list[str], list[str]]:
    """Load corpus documents with optional filtering.
    
    Args:
        path: Directory path containing corpus files
        **filters: Boolean flags for filtering:
            - remove_epistles: Remove epistolary texts
            - remove_test: Remove test document (Quaestio)
            - remove_egloghe: Remove eclogues 
            - remove_anonymus_files: Remove anonymous/misc texts
            - remove_unique_authors: Remove texts by authors with single work
            - remove_monarchia: Remove Monarchia text
    
    Returns:
        Tuple of (documents, authors, filenames)
    """
    files = [f for f in Path(path).glob('*.txt')]
    corpus = []
    
    for file in tqdm(files, desc=f'Loading corpus from {path}'):
        if _should_skip_file(file.name, filters):
            print(f'Removing {file.name}')
            continue
            
        author, title = file.stem.split('-')
        text = _clean_text(file.read_text(encoding='utf8', errors='ignore'))
        
        corpus.append({
            'text': text,
            'author': author.strip(),
            'filename': file.stem
        })

    if filters.get('remove_unique_authors'):
        corpus = _remove_single_author_texts(corpus)

    documents = [doc['text'] for doc in corpus]
    authors = [doc['author'] for doc in corpus]
    filenames = [doc['filename'] for doc in corpus]

    print(f'Total documents: {len(documents)}')
    print(f'Total authors: {len(set(authors))}')
    
    return documents, authors, filenames

def _should_skip_file(filename: str, filters: dict) -> bool:
    """Check if file should be filtered out based on criteria."""
    checks = {
        'remove_epistles': lambda f: 'epistola' in f.lower(),
        'remove_test': lambda f: 'apocrifo' in f.lower(),
        'remove_egloghe': lambda f: 'egloga' in f.lower(),
        'remove_anonymus_files': lambda f: any(x in f.lower() for x in ['misc', 'anonymus']),
        'remove_monarchia': lambda f: 'monarchia' in f.lower(),
        'remove_quijote': lambda f: 'don quijiote' in f.lower(),
    }
    return any(check(filename) for flag, check in checks.items() if filters.get(flag))

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








