import os
import pickle
import re
from pathlib import Path
from typing import List
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import unicodedata


# ------------------------------------------------------------------------
# document loading routine
# ------------------------------------------------------------------------
from data_preparation.segmentation import Segmentator


def get_spanish_function_words():
    from nltk.corpus import stopwords

    stop_words_sp = set(stopwords.words('spanish'))
    return stop_words_sp


def cache_idx_for_path(path):
    return Path(path).name.replace(' ', '_')


def cache_file_for_path(path, cache_path='./data_preparation/.cache'):
    return Path(cache_path) / f'processed_doc_{cache_idx_for_path(path)}.pkl'


def empty_cleaning_stats(original_chars=0):
    return {
        "original_chars": original_chars,
        "cleaned_chars": original_chars,
        "removed_spans": 0,
        "removed_chars": 0,
        "unmatched_open_brackets": 0,
        "unmatched_close_brackets": 0,
    }


def remove_square_bracketed_text(text, replacement=" "):
    output = []
    depth = 0
    removed_spans = 0
    removed_chars = 0
    unmatched_close = 0

    for character in text:
        if character == "[":
            if depth == 0:
                removed_spans += 1
                if replacement and (not output or not output[-1].isspace()):
                    output.append(replacement)
            depth += 1
            removed_chars += 1
            continue

        if character == "]":
            if depth > 0:
                depth -= 1
                removed_chars += 1
            else:
                unmatched_close += 1
                output.append(character)
            continue

        if depth > 0:
            removed_chars += 1
            continue

        output.append(character)

    return "".join(output), {
        "removed_spans": removed_spans,
        "removed_chars": removed_chars,
        "unmatched_open_brackets": depth,
        "unmatched_close_brackets": unmatched_close,
    }


def normalize_cleaned_text(text):
    text = text.replace('\x00', '')
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r"([(¿¡]) +", r"\1", text)

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not any(character.isalnum() for character in stripped):
            continue
        lines.append(line.rstrip())

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return f"{text}\n" if text else ""


class Book:

    def __init__(self, path, remove_square_bracketed=True):
        author, title = path.stem.split('-')
        raw_text = path.read_text(encoding='utf8', errors='ignore')
        self.remove_square_bracketed = remove_square_bracketed
        self.cleaning_stats = empty_cleaning_stats(len(raw_text))
        clean_text = self._clean_text(raw_text)
        author_normalized = self._normalize_author(author)

        self.path = path
        self.title = title.strip()
        self.author = author_normalized
        self.original_author = author_normalized
        self.raw_text = raw_text
        self.clean_text = clean_text
        self.processed = None
        self.segmented = None
        # self.fragments = None

    def _clean_text(self, text):
        """Clean and normalize text content."""
        # text = text.lower()
        original_chars = len(text)
        text = text.replace('\x00', '')
        stats = empty_cleaning_stats(original_chars)

        if self.remove_square_bracketed:
            text, removal_stats = remove_square_bracketed_text(text)
            stats.update(removal_stats)
            text = normalize_cleaned_text(text)
        else:
            text = text.strip()

        stats["cleaned_chars"] = len(text)
        self.cleaning_stats = stats
        return text

    def _normalize_author(self, author):
        author = author.strip()
        author_normalized = ''.join(
            c for c in unicodedata.normalize('NFKD', author)
            if not unicodedata.combining(c)
        )
        return author_normalized

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
            import spacy

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

    def process_document(self, document, cache_idx, refresh=False):
        if refresh and cache_idx in self.cache:
            del self.cache[cache_idx]
        if cache_idx not in self.cache:
            print(f'{cache_idx} not in cache')
            processed_doc = self.get_nlp()(document)
            self.cache[cache_idx] = processed_doc
            self.save_cache()
        processed_doc = self.cache[cache_idx]
        return processed_doc


def _job_open_book(file, cache_path='./data_preparation/.cache', refresh_cache=False):

    cache_idx = cache_idx_for_path(file)
    processor = DocumentProcessor(savecache=f'{cache_path}/processed_doc_{cache_idx}.pkl')
    segmentator = Segmentator()

    book = Book(file)

    # spacy processing of the full document
    book.processed = processor.process_document(
        book.clean_text,
        cache_idx,
        refresh=refresh_cache,
    )

    # segmentation
    book.segmented = segmentator.transform(book.processed)

    return book


def _resolve_max_workers(n_jobs):
    if n_jobs == -1:
        return multiprocessing.cpu_count()
    return max(1, int(n_jobs))


def load_corpus(path: str, cache_path='./data_preparation/.cache', n_jobs=1, refresh_cache=False):

    multiprocessing.set_start_method("spawn", force=True)

    paths = Path(path).glob('*.txt')
    with ProcessPoolExecutor(max_workers=_resolve_max_workers(n_jobs)) as executor:
        futures = {
            executor.submit(_job_open_book, p, cache_path, refresh_cache): p
            for p in paths
        }
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