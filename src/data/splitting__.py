from copy import copy

import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from helpers__ import tokenize


class Segmentation:

    SPLIT_POLICIES = ['by_endline', 'by_sentence']

    def __init__(self, split_policy='by_sentence', tokens_per_fragment=500, min_tokens=8, keep_full=True):
        assert split_policy in Segmentation.SPLIT_POLICIES, \
            f'unknown policy, valid ones are {Segmentation.SPLIT_POLICIES}'
        self.split_policy = split_policy
        #self.window_size = window_size
        self.tokens_per_fragment = tokens_per_fragment
        self.min_tokens = min_tokens
        self.keep_full = keep_full
        self.groups = None

    def fit(self, X, y):
        return self

    def transform(self, documents, authors):
        fragments = copy(documents) if self.keep_full else []
        authors_fragments = copy(authors) if self.keep_full else []
        groups = list(range(len(documents))) if self.keep_full else  []
        for i, text in tqdm(enumerate(documents), total=len(documents), desc='generating fragments'):
            if self.split_policy == 'by_endline':
                text_fragments = self._split_by_endline(text)
            elif self.split_policy == 'by_sentence':
                text_fragments = self._split_by_sentences(text)
            text_fragments = self._windows(text_fragments, tokens_per_fragment=self.tokens_per_fragment)
            fragments.extend(text_fragments)
            groups.extend([i] * len(text_fragments))
            if authors is not None:
                authors_fragments.extend([authors[i]] * len(text_fragments))

        self.groups = np.asarray(groups)

        return fragments, authors_fragments

    def fit_transform(self, documents, authors):
        return self.fit(documents, authors).transform(documents, authors)

    def _split_by_endline(self, text):
        return [t.strip() for t in text.split('\n') if t.strip()]

    def _split_by_sentences(self, text):
        sentences = [t.strip() for t in nltk.tokenize.sent_tokenize(text) if t.strip()]
        for i, sentence in enumerate(sentences):
            n_tokens = len(tokenize(sentence))
            if n_tokens < self.min_tokens:
                if i < len(sentences) - 1:
                    sentences[i + 1] = sentences[i] + ' ' + sentences[i + 1]
                else:
                    sentences[i - 1] = sentences[i - 1] + ' ' + sentences[i]
                sentences.pop(i)
        return sentences

    # def __windows(self, text_fragments, window_size):
    #     new_fragments = []
    #     nbatches = len(text_fragments) // window_size
    #     if len(text_fragments) % window_size > 0:
    #         nbatches += 1
    #     for i in range(nbatches):
    #         offset = i*window_size
    #         new_fragments.append(' '.join(text_fragments[offset:offset+window_size]))
    #     return new_fragments
    
    def __windows(self, text_fragments, tokens_per_fragment):
        new_fragments = []
        for fragment in text_fragments:
            if len(word_tokenize(fragment)) < tokens_per_fragment:
                if not new_fragments: 
                        new_fragments.append(fragment)
                else:
                    new_fragments[-1] += ' ' + fragment
            else:
                new_fragments.append(fragment)
        return new_fragments
    
    def _windows(self, text_fragments, tokens_per_fragment):
        new_fragments = []
        current_batch = ""

        for fragment in text_fragments:
            tokens_count = len(word_tokenize(fragment))
            
            if tokens_count >= tokens_per_fragment:
                new_fragments.append(fragment)

            elif tokens_count < tokens_per_fragment:
                current_batch += " " + fragment if current_batch else fragment
                if len(word_tokenize(current_batch)) >= tokens_per_fragment:
                    new_fragments.append(current_batch.strip())
                    current_batch= ""

        if current_batch: # se rimane un batch che non supera min tokens
            new_fragments.append(current_batch.strip())
            
        return new_fragments
            

if __name__ == '__main__':
    from data.data_loader import load_corpus

    path = '../../LatinCorpus/Progetto Quaestio'
    documents, authors, filenames = load_corpus('../../LatinCorpus')
    print(f'read {len(documents)} documents')
    print(f'#authors {len(set(authors))}')

    splitter = Segmentation(split_policy='by_sentence', window_size=5, min_tokens=8, keep_full=True)
    documents, authors = splitter.fit_transform(documents, authors)
    groups = splitter.groups
    print(len(documents))
    print(len(authors))
    print(len(groups))