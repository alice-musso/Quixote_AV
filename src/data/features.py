import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
from scipy.sparse import hstack, csr_matrix, issparse
import spacy
#spacy.load( 'la_core_web_lg')
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import Normalizer

from helpers__ import tokenize, get_function_words
import numpy as np
#from cltk.tag.pos import POSTag
from tqdm import tqdm
from functools import lru_cache
import pickle
from nltk import ngrams

from string import punctuation

# NLP = spacy.load('la_core_web_lg')
# NLP.max_length = 1364544


class DocumentProcessor:
    def __init__(self, language_model, savecache='.cache/processed_docs_data.pkl'):
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

    # def process_documents(self, documents):
    #     processed_docs = {}
    #     for doc_id, doc in enumerate(documents):
    #         if doc in self.cache:
    #             processed_docs[doc_id] = self.cache[doc]
    #         else:
    #             processed_doc = self.nlp(doc)
    #             self.cache[doc] = processed_doc
    #             processed_docs[doc_id] = processed_doc
    #     self.save_cache()
    #     return processed_docs.values()
            
    def process_documents(self, documents, filenames):
        processed_docs = []
        for filename, doc in zip(filenames, documents):
            if filename in self.cache:
                #print('document already in cache')
                processed_docs.append(self.cache[filename])
            else:
                processed_doc = self.nlp(doc)
                self.cache[filename] = processed_doc
                processed_docs.append(processed_doc)
        self.save_cache()
        return processed_docs 
    

class FeaturesDVEX:

    def __init__(self, function_words, **tfidf_kwargs):
        self.function_words = function_words
        self.tfidf_kwargs = tfidf_kwargs
        

    def __str__(self) -> str:
        return 'FeaturesDVEX'


    def fit(self, documents, y=None):
        distortions = self.distortion(documents)
        self.vectorizer = TfidfVectorizer(**self.tfidf_kwargs)
        self.vectorizer.fit(distortions)
        return self


    def transform(self, tokens, y=None):
        distortions = self.distortion(tokens)
        features = self.vectorizer.transform(distortions)
        features_num = features.shape[1]
        
        # print(f'Vectorizer: {FeaturesDVEX}')
        # print('Features:', features_num)
        return features
    

    def fit_transform(self, tokens, y=None):
        distortions = self.distortion(tokens)
        self.vectorizer = TfidfVectorizer(**self.tfidf_kwargs)
        features = self.vectorizer.fit_transform(distortions)
        features_num = features.shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return features
    

    # DV-EX text distortion method from Stamatatos_2018:
    # Every word not in function_words is masked by replacing each of its characters with an asterisk (*),
    # except first and last one.
    # Words of len 2 or 1 remain the same.
    def distortion(self, documents):

        def DVEX(token):
            if len(token) <= 2:
                return token
            return token[0] + ('*' * (len(token) - 2)) + token[-1]

        dis_texts = []
        for doc in documents:
            tokens = [str(token) for token in doc]
            dis_text = [token if token in self.function_words else DVEX(token) for token in tokens]
            # for token in tokens:
            #     if token in self.function_words:
            #         dis_text.append(token)
            #     else:
            #         dis_text.append(DVEX(token))
            dis_texts.append(' '.join(dis_text))

        return dis_texts
    

class FeaturesMendenhall:
    """
    Extract features as the frequency of the words' lengths used in the documents,
    following the idea behind Mendenhall's Characteristic Curve of Composition
    """
    def __init__(self,upto=25):
        self.upto = upto

    def __str__(self) -> str:
        return 'FeaturesMendenhall'

    def fit(self, documents, y=None):
        return self

    def transform(self, documents, y=None):
        features = []
        for doc in documents:
        #     processed_doc = NLP(doc)
            word_lengths = [len(str(token)) for token in doc]
        #word_lengths = [len(str(token)) for token in tokens]

        #word_lengths = [len(token) for token in tokenize(doc)]
            hist = np.histogram(word_lengths, bins=np.arange(1, self.upto), density=True)[0]
            distributuion = np.cumsum(hist)
            features.append(distributuion)

        # features_num = np.asarray(features).shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return np.asarray(features)

    def fit_transform(self, documents, y=None):
        return self.fit(documents).transform(documents)


class FeaturesSentenceLength:
    def __init__(self, upto=1000, language='spanish'):
        self.upto = upto
        self.language = language

    def __str__(self) -> str:
        return 'FeaturesSentenceLength'

    def fit(self, documents, y=None):
        return self

    def transform(self, documents, y=None):
        features = []
        for doc in documents:
            #preocessed_doc = NLP(doc)
            sentences = [str(sentence) for sentence in doc.sents]
            #sentences = [t.strip() for t in nltk.tokenize.sent_tokenize(doc, language=self.language) if t.strip()]
            sentence_lengths = []
            for sentence in doc.sents:
                sent_len = [len(str(token)) for token in sentence]
                sentence_lengths += sent_len
            #sentence_lengths = [len(str(token)) for NLP(sentence) in sentences for token in sentence]
            hist = np.histogram(sentence_lengths, bins=np.arange(1, self.upto), density=True)[0]
            distributuion = np.cumsum(hist)
            features.append(distributuion)

        features_num = np.asarray(features).shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return np.asarray(features)

    def fit_transform(self, documents, y=None):
        return self.fit(documents).transform(documents)


class FeaturesCharNGram:

    def __init__(self, n=3, sublinear_tf=False, norm='l1'):
        self.n = n
        self.sublinear_tf = sublinear_tf
        self.norm = norm
    
    def __str__(self) -> str:
        return 'FeaturesCharNGram'

    def fit(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(self.n, self.n), use_idf=False, norm=self.norm, min_df=3).fit(raw_documents)
        return self

    def transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        return self.vectorizer.transform(raw_documents)

    def fit_transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(self.n, self.n), use_idf=False, norm=self.norm, min_df=3)
        return self.vectorizer.fit_transform(raw_documents)


class FeaturesFunctionWords:

    def __init__(self, function_words, use_idf=False, sublinear_tf=False, norm='l1'):
        # assert language in {'latin', 'spanish'}, 'the requested language is not yet covered'
        # self.language = languageƒ
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.function_words=function_words
    
    def __str__(self) -> str:
        return 'FeaturesFunctionWords'

    def fit(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        #function_words = get_function_words(self.language)
        self.vectorizer = TfidfVectorizer(
            vocabulary=self.function_words, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm)
        self.vectorizer.fit(raw_documents)
        return self

    def transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        return self.vectorizer.transform(raw_documents)

    def fit_transform(self, documents, y=None):
        #function_words = get_function_words(self.language)
        raw_documents = [doc.text for doc in documents]
        self.vectorizer = TfidfVectorizer(
            vocabulary=self.function_words, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm)
        
        features = self.vectorizer.fit_transform(raw_documents)
        # features_num =features.shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return features
    

class FeaturesPunctuation:

    def __init__(self, sublinear_tf=False, norm='l1'):
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.punctuation=punctuation
        # self.punctuation = "¡!\"#$%&'()*+,-./:;<=>¿?@[\\]^_`{|}~"
    
    def __str__(self) -> str:
        return 'FeaturesPunctuation'

    def fit(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]

        self.vectorizer = TfidfVectorizer(analyzer='char', vocabulary=self.punctuation, use_idf=False, norm=self.norm, min_df=3)
        self.vectorizer.fit(raw_documents)
        return self

    def transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        return self.vectorizer.transform(raw_documents)

    def fit_transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        
        self.vectorizer = TfidfVectorizer(analyzer='char', vocabulary=self.punctuation, use_idf=False, norm=self.norm, min_df=3)

        # features_num =features.shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return self.vectorizer.fit_transform(raw_documents)


# class FeaturesPOST:
#     def __init__(self, language, use_idf=True, sublinear_tf=True, norm='l2', savecache='.postcache/dict.pkl', **tfidf_kwargs):
#         #assert language in {'latin', 'spanish'}, 'the requested language is not yet covered'
#         if language == 'latin':
#             language = 'lat'
#         self.language = language
#         self.use_idf = use_idf
#         self.sublinear_tf = sublinear_tf
#         self.norm = norm
#         self.tfidf_kwargs = tfidf_kwargs
#         self.savecache = savecache
#         self.init_cache()
    
#     def __str__(self) -> str:
#         return 'FeaturesPOST'

#     def init_cache(self):
#         self.changed = False
#         if self.savecache is None or not os.path.exists(self.savecache):
#             print('cache not found, initializing from scratch')
#             self.cache = {}
#         else:
#             print(f'loading cache from {self.savecache}')
#             self.cache = pickle.load(open(self.savecache, 'rb'))

#     def save_cache(self):
#         if self.savecache is not None and self.changed:
#             print(f'storing POST cache in {self.savecache}')
#             parent = Path(self.savecache).parent
#             if parent:
#                 os.makedirs(parent, exist_ok=True)
#             pickle.dump(self.cache, open(self.savecache, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
#             self.changed = False

#     def post_analyzer(self, text):
#         ngram_range = self.tfidf_kwargs.get('ngram_range', (1,4)) # up to quadrigrams
#         ngram_range = slice(*ngram_range)
#         tags = []
#         for sentence in nltk.tokenize.sent_tokenize(text):
#             sentence_wordtags = self.get_postags(sentence)
#             sentence_unigram_tags = [tag[0] if tag != 'Unk' else 'Unk' for _, tag in sentence_wordtags] #prima tag[0]
#             for n in list(range(ngram_range.start, ngram_range.stop+1)):
#                 sentence_ngram_tags = ['-'.join(ngram) for ngram in list(ngrams(sentence_unigram_tags, n))]
#                 tags.extend(sentence_ngram_tags)
#         # print('+', end='')
#         return tags

#     def get_postags(self, sentence):
#         if sentence in self.cache:
#             tags = self.cache[sentence]
#         else:
#             #tags = self.tagger.tag_tnt(sentence) # sostituire con spicy
#             processed_sent = self.tagger(sentence)
#             tags = [(str(token), token.pos_)  for token in processed_sent]
#             self.cache[sentence] = tags
#             self.changed = True
#         return tags

#     def fit(self, documents, y=None):
#         #self.tagger = POSTag(self.language) # sostituire con spacy
#         self.tagger = spacy.load( 'es_core_news_md')
#         self.vectorizer = TfidfVectorizer(
#             analyzer=self.post_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
#         self.vectorizer.fit(documents)
#         self.save_cache()
#         return self

#     def transform(self, documents, y=None):
#         post_features = self.vectorizer.transform(documents)
#         self.save_cache()
#         return post_features

#     def fit_transform(self, documents, y=None):
#         #self.tagger = POSTag(self.language) # sostituire con spicy
#         self.tagger = spacy.load( 'es_core_news_md')
#         self.vectorizer = TfidfVectorizer(
#             analyzer=self.post_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
#         post_features = self.vectorizer.fit_transform(documents)
#         self.save_cache()
#         return post_features
    

# class FeaturesDEP:
#     def __init__(self, use_idf=True, sublinear_tf=True, norm='l2', savecache='.depcache/dict.pkl', **tfidf_kwargs):
#         self.use_idf = use_idf
#         self.sublinear_tf = sublinear_tf
#         self.norm = norm
#         self.tfidf_kwargs = tfidf_kwargs
#         self.savecache = savecache
#         self.init_cache()
    
#     def __str__(self) -> str:
#         return 'FeaturesDEP'

#     def init_cache(self):
#         self.changed = False
#         if self.savecache is None or not os.path.exists(self.savecache):
#             print('cache not found, initializing from scratch')
#             self.cache = {}
#         else:
#             print(f'loading cache from {self.savecache}')
#             self.cache = pickle.load(open(self.savecache, 'rb'))

#     def save_cache(self):
#         if self.savecache is not None and self.changed:
#             print(f'storing DEP cache in {self.savecache}')
#             parent = Path(self.savecache).parent
#             if parent:
#                 os.makedirs(parent, exist_ok=True)
#             pickle.dump(self.cache, open(self.savecache, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
#             self.changed = False

#     def dep_analyzer(self, text):
#         ngram_range = self.tfidf_kwargs.get('ngram_range', (2,3))
#         ngram_range = slice(*ngram_range)
#         deps = []
#         for sentence in nltk.tokenize.sent_tokenize(text):
#             sentence_deps = self.get_dependencies(sentence)
#             sentence_unigram_deps = [dep[0] for _, dep in sentence_deps] #prima tag[0]
#             for n in list(range(ngram_range.start, ngram_range.stop+1)):
#                 sentence_ngram_deps = ['-'.join(ngram) for ngram in list(ngrams(sentence_unigram_deps, n))]
#                 deps.extend(sentence_ngram_deps)
#         # print('+', end='')
#         return deps

#     def get_dependencies(self, sentence):
#         if sentence in self.cache:
#             dependencies = self.cache[sentence]
#         else:
#             #tags = self.tagger.tag_tnt(sentence) # sostituire con spicy
#             processed_sent = self.tagger(sentence)
#             dependencies = [(str(token), token.dep_)  for token in processed_sent]
#             self.cache[sentence] = dependencies
#             self.changed = True
#         return dependencies

#     def fit(self, documents, y=None):
#         #self.tagger = POSTag(self.language) # sostituire con spacy
#         self.tagger = spacy.load( 'es_core_news_md')
#         self.vectorizer = TfidfVectorizer(
#             analyzer=self.dep_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
#         self.vectorizer.fit(documents)
#         self.save_cache()
#         return self

#     def transform(self, documents, y=None):
#         dep_features = self.vectorizer.transform(documents)
#         self.save_cache()
#         return dep_features

#     def fit_transform(self, documents, y=None):
#         #self.tagger = POSTag(self.language) # sostituire con spicy
#         self.tagger = spacy.load( 'es_core_news_md')
#         self.vectorizer = TfidfVectorizer(
#             analyzer=self.dep_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
#         dep_features = self.vectorizer.fit_transform(documents)
#         self.save_cache()
#         return dep_features
    

class FeatureSetReductor:
    def __init__(self, feature_extractor, measure=chi2, k=100, k_ratio=1.0, normalize=True):
        self.feature_extractor = feature_extractor
        self.k = k
        self.k_ratio = k_ratio
        self.measure = measure
        #self.feat_sel = SelectKBest(measure, k=self.k)

        self.normalize = normalize #new!
        if self.normalize:
            self.normalizer = Normalizer()
    
    def __str__(self) -> str:
        return( f'FeatureSetReductor for {self.feature_extractor}' )


    def fit(self, documents, authors=None):
        matrix = self.feature_extractor.fit_transform(documents, authors)
        features_in = matrix.shape[1]

        if features_in < 100:
            self.k = features_in
            self.feat_sel = SelectKBest(self.measure, k='all')
        else:
            #self.k = round(features_in * 0.1) #keep 10% of features
            self.k = round(features_in * self.k_ratio) #keep k_ratio% of features
            self.feat_sel = SelectKBest(self.measure, k=self.k)

        if self.normalize:
            self.normalizer.fit(matrix)

        self.feat_sel.fit(matrix, authors)

        print(self)
        print('features in:', features_in, 'k:', self.k)
        print()
        return self

    def transform(self, documents, authors=None):
        matrix = self.feature_extractor.transform(documents)
        if self.normalize:
            matrix_norm  = self.normalizer.transform(matrix)
            matrix_red = self.feat_sel.transform(matrix_norm)
        else:
            matrix_red = self.feat_sel.transform(matrix)
        return matrix_red #self.feat_sel.transform(matrix)

    def fit_transform(self, documents, authors=None):
        return self.fit(documents,authors).transform(documents, authors)
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
            
            if key == 'measure':
                self.feat_sel = SelectKBest(value, k=self.k)
            elif key == 'k_ratio':
                self.feat_sel.k_ratio = value
            elif key == 'normalize':
                if value:
                    self.normalizer = Normalizer()
                else:
                    self.normalizer = None
        return self
    
    def get_params(self, deep=True):
        params = {
            'feature_extractor': self.feature_extractor,
            'measure': self.measure,
            'k': self.k,
            'normalize': self.normalize
        }
        return params
    



class HstackFeatureSet:
    def __init__(self, feats=None, *vectorizers):
        self.vectorizers = vectorizers

    def fit(self, documents, authors=None):
        for v in self.vectorizers:
            v.fit(documents, authors)
        return self

    def transform(self, documents, authors=None):
        feats = [v.transform(documents, authors) for v in self.vectorizers]
        return self._hstack(feats)

    def fit_transform(self, documents, authors=None):
        feats = [v.fit_transform(documents, authors) for v in self.vectorizers]
        return self._hstack(feats)

    def _hstack(self, feats):
        anysparse = any(map(issparse, feats))
        if anysparse:
            feats = [csr_matrix(f) for f in feats]
            feats = hstack(feats)
        else:
            feats = np.hstack(feats)
        return feats


class FeaturesPOST:
    def __init__(self, use_idf=True, sublinear_tf=True, norm='l2', savecache='.postcache/dict.pkl', **tfidf_kwargs):
        #assert language in {'latin', 'spanish'}, 'the requested language is not yet covered'
        # if language == 'latin':
        #     language = 'lat'
        # self.language = language
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.tfidf_kwargs = tfidf_kwargs
        self.savecache = savecache
        #self.tagger=spacy.load('la_core_web_lg')
        # self.init_cache()
    
    def __str__(self) -> str:
        return 'FeaturesPOST'

    # def init_cache(self):
    #     self.changed = False
    #     if self.savecache is None or not os.path.exists(self.savecache):
    #         print('cache not found, initializing from scratch')
    #         self.cache = {}
    #     else:
    #         print(f'loading cache from {self.savecache}')
    #         self.cache = pickle.load(open(self.savecache, 'rb'))

    # def save_cache(self):
    #     if self.savecache is not None and self.changed:
    #         print(f'storing POST cache in {self.savecache}')
    #         parent = Path(self.savecache).parent
    #         if parent:
    #             os.makedirs(parent, exist_ok=True)
    #         pickle.dump(self.cache, open(self.savecache, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    #         self.changed = False

    def post_analyzer(self, doc):
        ngram_range = self.tfidf_kwargs.get('ngram_range', (1,4)) # up to quadrigrams
        ngram_range = slice(*ngram_range)
        ngram_tags = []
        #processed_doc = NLP(text)
        for sentence in doc.sents:
            sentence_unigram_tags = [token.pos_[0] if token.pos_ != '' else 'Unk' for token in sentence]
            #sentence_wordtags = self.get_postags(str(sentence))
            #try:
            #   sentence_unigram_tags = [tag[0] if tag != '' else 'Unk' for _, tag in sentence_wordtags] 
            # except IndexError:
            #     print(sentence_wordtags)
            #     assert False

            for n in list(range(ngram_range.start, ngram_range.stop+1)):
                sentence_ngram_tags = ['-'.join(ngram) for ngram in list(ngrams(sentence_unigram_tags, n))]
                ngram_tags.extend(sentence_ngram_tags)
        # print('+', end='')
        return ngram_tags

    # def get_postags(self, sentence):
    #     if sentence in self.cache:
    #         tags = self.cache[sentence]
    #     else:
    #         #tags = self.tagger.tag_tnt(sentence) # sostituire con spicy
    #         processed_sent = self.tagger(sentence)
    #         tags = [(str(token), token.pos_)  for token in processed_sent]
    #         self.cache[sentence] = tags
    #         self.changed = True
    #     return tags

    def fit(self, documents, y=None):
        #self.tagger = POSTag(self.language) 
        #self.tagger = spacy.load('la_core_web_lg')
        self.vectorizer = TfidfVectorizer(
            analyzer=self.post_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
        self.vectorizer.fit(documents)
        # self.save_cache()
        return self

    def transform(self, documents, y=None):
        post_features = self.vectorizer.transform(documents)
        # self.save_cache()

        # features_num =post_features.shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return post_features

    def fit_transform(self, documents, y=None):
        #self.tagger = POSTag(self.language) # sostituire con spicy
        #self.tagger = spacy.load('la_core_web_lg')
        self.vectorizer = TfidfVectorizer(
            analyzer=self.post_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
        post_features = self.vectorizer.fit_transform(documents)
        # self.save_cache()

        # features_num = post_features.shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return post_features
    

class FeaturesDEP:
    def __init__(self, use_idf=True, sublinear_tf=True, norm='l2', savecache='.depcache/dict.pkl', **tfidf_kwargs):
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.tfidf_kwargs = tfidf_kwargs
        self.savecache = savecache
        # self.init_cache()
    
    def __str__(self) -> str:
        return 'FeaturesDEP'

    # def init_cache(self):
    #     self.changed = False
    #     if self.savecache is None or not os.path.exists(self.savecache):
    #         print('cache not found, initializing from scratch')
    #         self.cache = {}
    #     else:
    #         print(f'loading cache from {self.savecache}')
    #         self.cache = pickle.load(open(self.savecache, 'rb'))

    # def save_cache(self):
    #     if self.savecache is not None and self.changed:
    #         print(f'storing DEP cache in {self.savecache}')
    #         parent = Path(self.savecache).parent
    #         if parent:
    #             os.makedirs(parent, exist_ok=True)
    #         pickle.dump(self.cache, open(self.savecache, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    #         self.changed = False

    def dep_analyzer(self, doc):
        ngram_range = self.tfidf_kwargs.get('ngram_range', (2,3))
        ngram_range = slice(*ngram_range)
        ngram_deps = []
        # processed_doc = NLP(text)
        for sentence in doc.sents:
        # #for sentence in nltk.tokenize.sent_tokenize(text):
            sentence_unigram_deps = [token.dep_[0] if token.dep_ != '' else 'Unk' for token in sentence]
        #     sentence_unigram_deps = [dep[0] if dep != '' else 'Unk' for _, dep in sentence_deps] #prima tag[0]

            for n in list(range(ngram_range.start, ngram_range.stop+1)):
                sentence_ngram_deps = ['-'.join(ngram) for ngram in list(ngrams(sentence_unigram_deps, n))]
                ngram_deps.extend(sentence_ngram_deps)
        # print('+', end='')
        return ngram_deps


    # def get_dependencies(self, sentence):
    #     if sentence in self.cache:
    #         dependencies = self.cache[sentence]
    #     else:
    #         #tags = self.tagger.tag_tnt(sentence) # sostituire con spicy
    #         processed_sent = self.tagger(sentence)
    #         dependencies = [(str(token), token.dep_)  for token in processed_sent]
    #         self.cache[sentence] = dependencies
    #         self.changed = True
    #     return dependencies

    def fit(self, documents, y=None):
        #self.tagger = POSTag(self.language) # sostituire con spacy
        #self.tagger = spacy.load('la_core_web_lg')
        self.vectorizer = TfidfVectorizer(
            analyzer=self.dep_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
        self.vectorizer.fit(documents)
        #self.save_cache()
        return self

    def transform(self, documents, y=None):
        dep_features = self.vectorizer.transform(documents)
        #self.save_cache()

        features_num =dep_features.shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return dep_features

    def fit_transform(self, documents, y=None):
        #self.tagger = POSTag(self.language) # sostituire con spicy
        #self.tagger = spacy.load( 'la_core_web_lg')
        self.vectorizer = TfidfVectorizer(
            analyzer=self.dep_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
        dep_features = self.vectorizer.fit_transform(documents)
        #self.save_cache()

        # features_num =dep_features.shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return dep_features