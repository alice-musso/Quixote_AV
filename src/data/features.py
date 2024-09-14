import os
from pathlib import Path
import re

from sklearn.exceptions import NotFittedError
from data_loader import clean_texts
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from pydro.src.dro import DistributionalRandomOversampling 
from dro import DistributionalRandomOversampling
from sklearn.utils.validation import check_is_fitted
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
from cltk.prosody.lat.macronizer import Macronizer
from cltk.prosody.lat.scanner import Scansion
from nltk import word_tokenize, sent_tokenize

from string import punctuation

# NLP = spacy.load('la_core_web_lg')
# NLP.max_length = 1364544


class DocumentProcessor:
    def __init__(self, language_model=None, savecache='.cache/processed_docs_cleaned.pkl'):
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

    def delete_doc(self, filename):
        removed_doc = self.cache.pop(filename, None)
        if removed_doc is not None:
            print(f'Removed {filename} from cache')
            self.save_cache()  # Salva la cache aggiornata dopo l'eliminazione
        
        else:
            print(f'{filename} not found in cache')
            

    # def _process_documents(self, documents, filenames):
    #     processed_docs = []
    #     for filename, doc in zip(filenames, documents):
    #         if filename in self.cache:
    #             #print('document already in cache')
    #             processed_docs.append(self.cache[filename])
    #         else:
    #             print(f'{filename} not in cache')
    #             processed_doc = self.nlp(doc)
    #             self.cache[filename] = processed_doc
    #             processed_docs.append(processed_doc)
    #             self.save_cache()
    #     return processed_docs 
            
    def process_documents(self, documents, filenames):
        processed_docs = {}
        for filename, doc in zip(filenames, documents):
            if filename in self.cache:
                #print('document already in cache')
                processed_docs[filename[:-2]] = self.cache[filename]
            else:
                print(f'{filename} not in cache')
                processed_doc = self.nlp(doc)
                self.cache[filename] = processed_doc
                processed_docs[filename[:-2]] = self.cache[filename]
                self.save_cache()
        return processed_docs 
    

class FeaturesDistortedView:

    def __init__(self, function_words, method, ngram_range=(1,1), **tfidf_kwargs):
        assert method in {'DVEX', 'DVMA', 'DVSA'}, 'text distortion method not valid'
        self.function_words = function_words
        self.ngram_range = ngram_range
        self.tfidf_kwargs = tfidf_kwargs
        self.method = method
        self.counter = CountVectorizer()
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, **self.tfidf_kwargs)
        self.training_words = []

    def __str__(self) -> str:
        ngram_range_str = f' [n-gram range: {self.ngram_range}]'
        if self.method=='DVEX':
            return 'FeaturesDVEX'+ ngram_range_str
        if self.method=='DVMA':
            return 'FeaturesDVMA'+ ngram_range_str
        if self.method=='DVSA':
            return 'FeaturesDVSA'+ ngram_range_str


    def fit(self, documents, y=None):
        distortions = self.distortion(documents, method=self.method)
        self.vectorizer.fit(distortions)
        return self


    def transform(self, documents, y=None):
        distortions = self.distortion(documents, method=self.method)
        self.count_words(distortions)
        features = self.vectorizer.transform(distortions)
        features_num = features.shape[1]
        # print(f'Vectorizer: {FeaturesDVEX}')
        # print('Features:', features_num)
        return features
    

    def fit_transform(self, documents, y=None):
        distortions = self.distortion(documents, method=self.method)
        self.count_words(distortions)
        features = self.vectorizer.fit_transform(distortions)
        # features_num = features.shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return features
    
    def distortion(self, documents, method):
        if method == 'DVEX':
            dis_texts = self.dis_DVEX(documents)
        elif method =='DVMA':
            dis_texts = self.dis_DVMA(documents)
        elif method =='DVSA':
            dis_texts = self.dis_DVSA(documents)
        return dis_texts
    
    def count_words(self, texts):
        if not hasattr(self, 'n_training_terms'):
           self.training_words = self.counter.fit_transform(texts) 
           self.n_training_terms = self.training_words.sum(axis=1).getA().flatten()
        else:
            self.test_words = self.counter.transform(texts)
            self.n_test_terms = self.test_words.sum(axis=1).getA().flatten()



    # DV-MA text distortion method from Stamatatos_2018:
    # Every word not in function_words is masked by replacing each of its characters with an asterisk (*).
    # for character embedding
    def dis_DVMA(self, docs):
        dis_texts = []
        for doc in tqdm(docs, 'DV-MA distorting', total=len(docs)):
            tokens = [str(token) for token in doc]
            dis_text = ''
            for token in tokens:
                if dis_text != '' and token != '.':
                    dis_text += ' '
                if token in self.function_words or token == '.':
                    dis_text += token
                else:
                    dis_text += '*' * len(token)
            dis_texts.append(dis_text)
        return dis_texts
    
    
    # DV-SA text distortion method from Stamatatos_2018:
    # Every word not in function_words is replaced with an asterisk (*).
    # for character embedding
    def dis_DVSA(self, docs):
        dis_texts = []
        for doc in tqdm(docs, 'DV-SA distorting', total=len(docs)):
            tokens = [str(token) for token in doc]
            dis_text = ''
            for token in tokens:
                if dis_text != '' and token != '.':
                    dis_text += ' '
                if token in self.function_words or token == '.':
                    dis_text += token
                else:
                    dis_text += '*'
            dis_texts.append(dis_text)
        return dis_texts
    

    # DV-EX text distortion method from Stamatatos_2018:
    # Every word not in function_words is masked by replacing each of its characters with an asterisk (*),
    # except first and last one.
    # Words of len 2 or 1 remain the same.
    def dis_DVEX(self, documents):

        def DVEX(token):
            if len(token) <= 2:
                return token
            return token[0] + ('*' * (len(token) - 2)) + token[-1]

        dis_texts = []
        for doc in tqdm(documents, 'DV-EX distorting', total=len(documents)):
            tokens = [str(token) for token in doc]
            dis_text = [token if token in self.function_words else DVEX(token) for token in tokens]
            # for token in tokens:
            #     if token in self.function_words:
            #         dis_text.append(token)
            #     else:
            #         dis_text.append(DVEX(token))
            dis_texts.append(' '.join(dis_text))

        return dis_texts
    
        

class FeaturesSyllabicQuantities:

    def __init__(self, min_range=1,max_range=1, ngram_range=(1,1), **tfidf_kwargs):
        self.tfidf_kwargs = tfidf_kwargs
        self.min_range = min_range
        self.max_range = max_range
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, **self.tfidf_kwargs)
        self.counter = CountVectorizer()
        

    def __str__(self) -> str:
        ngram_range_str = f' [n-gram range: {self.ngram_range}]'
        return 'FeaturesSyllabicQuantities' + ngram_range_str


    def fit(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        scanned_texts = self.metric_scansion(documents)
        #self.count_syllabic_quantities(scanned_texts)
        #self.vectorizer = TfidfVectorizer(**self.tfidf_kwargs)
        self.vectorizer.fit(scanned_texts)
        return self


    def transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        scanned_texts = self.metric_scansion(documents)
        self.count_syllabic_quantities(scanned_texts)
        features = self.vectorizer.transform(scanned_texts)
        return features
    

    def fit_transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        scanned_texts = self.metric_scansion(documents)
        self.count_syllabic_quantities(scanned_texts)
        # self.vectorizer = TfidfVectorizer(**self.tfidf_kwargs)
        features = self.vectorizer.fit_transform(scanned_texts)
        return features
    

    def metric_scansion(self, documents, filenames=None):
        #documents = [self.remove_invalid_word(doc, filename) for doc, filename in zip(documents, filenames)]
        documents = [self.remove_invalid_word(doc) for doc in documents]
            
        macronizer = Macronizer('tag_ngram_123_backoff')
        scanner = Scansion(
            clausula_length=100000, punctuation=string.punctuation)  # clausula_length was 13, it didn't get the string before that point (it goes backward)
        macronized_texts = [macronizer.macronize_text(doc) for doc in tqdm(documents, 'macronizing', total=len(documents))]
        scanned_texts = [scanner.scan_text(doc) for doc in
                        tqdm(macronized_texts, 'metric scansion', total=len(macronized_texts))]
        scanned_texts = [''.join(scanned_text) for scanned_text in scanned_texts]  # concatenate the sentences
        return scanned_texts
    
    def remove_invalid_word(self, document, filename=None):
        # todo: salvare i numeri romani, i numeri
        legal_words=[]
        vowels = set('aeiouāēīōū')
        tokens = [token.text for token in document]
        illegal_tokens=[]

        for token in tokens:
            token = token.lstrip()
            if len(token) == 1:
                if token.lower() in vowels or token in punctuation:
                    legal_words.append(token)
            elif len(token) == 2:
                if not all(char in punctuation for char in token) and not all(char not in vowels for char in token): 
                    legal_words.append(token)
            else:
                if (
                    any(char in vowels for char in token)
                    and not any(
                        token[i] in punctuation and token[i + 1] in punctuation
                        for i in range(len(token) - 1)
                    )
                ):
                    legal_words.append(token)

            if token not in legal_words:
                illegal_tokens.append(token)
        
        if filename:

            with open("illegal_words.txt", "a") as file:
                file.write(f"{filename}\n")
                file.write(f"{str(document)[:50]}\n")
                file.write(f"{illegal_tokens}\n")
                file.write("\n")
                

        return ' '.join(legal_words)


    def count_syllabic_quantities(self, texts):
        if not hasattr(self, 'n_training_terms'):
            self.training_words = self.counter.fit_transform(texts) 
            self.n_training_terms = self.training_words.sum(axis=1).getA().flatten()
        else:
            self.test_words = self.counter.transform(texts)
            self.n_test_terms = self.test_words.sum(axis=1).getA().flatten()
    

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
        for doc in tqdm(documents, 'Extracting word lenghts', total=len(documents)):
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
        for doc in tqdm(documents, 'Extracting sentence lenghts', total=len(documents)):
            #preocessed_doc = NLP(doc)
            #sentences = [str(sentence) for sentence in doc.sents]
            #sentences = [t.strip() for t in nltk.tokenize.sent_tokenize(doc, language=self.language) if t.strip()]
            sentence_lengths = []
            for sentence in doc.sents:
                sent_len = [len(str(token)) for token in sentence]
                sentence_lengths += sent_len
            #sentence_lengths = [len(str(token)) for NLP(sentence) in sentences for token in sentence]
            hist = np.histogram(sentence_lengths, bins=np.arange(1, self.upto), density=True)[0]
            distributuion = np.cumsum(hist)
            features.append(distributuion)

        # features_num = np.asarray(features).shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return np.asarray(features)

    def fit_transform(self, documents, y=None):
        return self.fit(documents).transform(documents)


class FeaturesCharNGram:

    def __init__(self, n=(1,3), sublinear_tf=False, norm='l1'):
        self.n = n
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.counter = CountVectorizer(analyzer='char', ngram_range=self.n, min_df=3)
    
    def __str__(self) -> str:
        return f'FeaturesCharNGram [n-gram range: ({self.n[0]},{self.n[1]})]'

    def fit(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(self.n), use_idf=False, norm=self.norm, min_df=3).fit(raw_documents)
        return self

    def transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.count_ngrams(raw_documents)
        return self.vectorizer.transform(raw_documents)

    def fit_transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.count_ngrams(raw_documents)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(self.n), use_idf=False, norm=self.norm, min_df=3)
        return self.vectorizer.fit_transform(raw_documents)

    
    def count_ngrams(self, texts):
        #raw_texts = [doc.text for doc in texts]
        if not hasattr(self, 'n_training_terms'):
            self.training_ngrams = self.counter.fit_transform(texts)
            self.n_training_terms = self.training_ngrams.sum(axis=1).getA().flatten()
        else:
            # Trasforma i nuovi testi e calcola il numero di n-grams
            self.test_ngrams = self.counter.transform(texts)
            self.n_test_terms = self.test_ngrams.sum(axis=1).getA().flatten()


class FeaturesFunctionWords:

    def __init__(self, function_words, use_idf=False, sublinear_tf=False, norm='l1', ngram_range=(1,3)):
        # assert language in {'latin', 'spanish'}, 'the requested language is not yet covered'
        # self.language = language
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.function_words=function_words
        self.ngram_range=ngram_range
        self.counter = CountVectorizer(vocabulary=self.function_words, min_df=1)
    
    def __str__(self) -> str:
        ngram_range_str = f' [n-gram range: {self.ngram_range}]'
        return 'FeaturesFunctionWords' + ngram_range_str

    def fit(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        
        #function_words = get_function_words(self.language)
        self.vectorizer = TfidfVectorizer(
            vocabulary=self.function_words, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, ngram_range=self.ngram_range)
        self.vectorizer.fit(raw_documents)
        return self

    def transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.count_words(raw_documents)
        return self.vectorizer.transform(raw_documents)

    def fit_transform(self, documents, y=None):
        #function_words = get_function_words(self.language)
        raw_documents = [doc.text for doc in documents]
        self.count_words(raw_documents)
        self.vectorizer = TfidfVectorizer(
            vocabulary=self.function_words, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm)
        
        features = self.vectorizer.fit_transform(raw_documents)
        # features_num =features.shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return features
    
    def count_words(self, texts):
        if not hasattr(self, 'n_training_terms'):
            self.training_words = self.counter.fit_transform(texts)
            self.n_training_terms = self.training_words.sum(axis=1).getA().flatten()
        else:
            # Trasforma i nuovi testi e calcola il numero di n-grams
            self.test_words = self.counter.transform(texts)
            self.n_test_terms = self.test_words.sum(axis=1).getA().flatten()
    

class FeaturesPunctuation:

    def __init__(self, sublinear_tf=False, norm='l1', ngram_range=(1,3)):
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.punctuation=punctuation
        self.ngram_range = ngram_range
        self.counter = CountVectorizer(vocabulary=self.punctuation, min_df=1)
        # self.punctuation = "¡!\"#$%&'()*+,-./:;<=>¿?@[\\]^_`{|}~"
    
    def __str__(self) -> str:
        ngram_range_str = f' [n-gram range: {self.ngram_range}]'
        return 'FeaturesPunctuation' + ngram_range_str

    def fit(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]

        self.vectorizer = TfidfVectorizer(analyzer='char', vocabulary=self.punctuation, use_idf=False, norm=self.norm, min_df=3, ngram_range=self.ngram_range)
        self.vectorizer.fit(raw_documents)
        return self

    def transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.count_words(raw_documents)
        return self.vectorizer.transform(raw_documents)

    def fit_transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.count_words(raw_documents)
        
        self.vectorizer = TfidfVectorizer(analyzer='char', vocabulary=self.punctuation, use_idf=False, norm=self.norm, min_df=3, ngram_range=(1,1))

        # features_num =features.shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return self.vectorizer.fit_transform(raw_documents)

    def count_words(self, texts):
        if not hasattr(self, 'n_training_terms'):
            self.training_words = self.counter.fit_transform(texts)
            self.n_training_terms = self.training_words.sum(axis=1).getA().flatten()
        else:
            # Trasforma i nuovi testi e calcola il numero di n-grams
            self.test_words = self.counter.transform(texts)
            self.n_test_terms = self.test_words.sum(axis=1).getA().flatten()



    

class FeatureSetReductor:
    def __init__(self, feature_extractor, measure=chi2, k=100, k_ratio=1.0, normalize=True, oversample=True):
        self.feature_extractor = feature_extractor
        self.k = k
        self.k_ratio = k_ratio
        self.measure = measure
        self.normalize = normalize 
        self.oversample = oversample
        self.is_sparse = True
        if self.normalize:
            self.normalizer = Normalizer()
        
    def __str__(self) -> str:
        return( f'FeatureSetReductor for {self.feature_extractor}' )


    def fit(self, documents, y_dev=None):
        return self.feature_extractor.fit(documents, y_dev)

    def transform(self, documents, y_dev=None):
        matrix = self.feature_extractor.transform(documents)

        if self.normalize:
            matrix_norm  = self.normalizer.transform(matrix) 
            matrix_red = self.feat_sel.transform(matrix_norm)
        else:
            matrix_red = self.feat_sel.transform(matrix, y_dev)
        return matrix_red #self.feat_sel.transform(matrix)

    def fit_transform(self, documents, y_dev=None):
        matrix = self.feature_extractor.fit_transform(documents, y_dev)
        features_in = matrix.shape[1]

        if features_in < self.k:
            self.k = features_in
        else:
            #self.k = round(features_in * 0.1) #keep 10% of features
            self.k = round(features_in * self.k_ratio) #keep k_ratio% of features

        self.feat_sel = SelectKBest(self.measure, k=self.k)

        #print(self)
        print('features in:', features_in, 'k:', self.k)
        print()

        if self.normalize:
            matrix_norm  = self.normalizer.fit_transform(matrix, y_dev)
            matrix_red = self.feat_sel.fit_transform(matrix_norm, y_dev)
            
        else:
            matrix_red = self.feat_sel.fit_transform(matrix, y_dev)

        return matrix_red
    
    def oversample_DRO(self, Xtr, ytr, Xte):
        if not isinstance(ytr, np.ndarray):
            ytr = np.array(ytr)
        self.dro = DistributionalRandomOversampling(rebalance_ratio=0.2)
        samples = self.dro._samples_to_match_ratio(ytr)
        y_oversampled = self.dro._oversampling_observed(ytr, samples)
        y_examples_generated = y_oversampled[len(ytr):]

        n_examples = samples.sum() - len(ytr)

        if hasattr(self.feature_extractor, 'n_training_terms'):
            print('Oversampling positive class using DRO method')

            #self.dro = DistributionalRandomOversampling(rebalance_ratio=0.2)
            self.n_training_terms =  self.feature_extractor.n_training_terms
            self.n_test_terms = self.feature_extractor.n_test_terms

            # print('Training terms: ', self.n_training_terms)
            # print('Test terms: ', self.n_test_terms)

            positives = ytr.sum()
            nD = len(ytr) 

            print('Before oversampling')
            print(f'positives = {positives} (prevalence={positives*100/nD:.2f}%)')

            Xtr, ytr = self.dro.fit_transform(Xtr, ytr, self.n_training_terms)
            Xte = self.dro.transform(Xte, self.n_test_terms, samples=1)

            positives = ytr.sum()
            nD = len(ytr)
            print('After oversampling')
            print(f'positives = {positives} (prevalence={positives*100/nD:.2f}%)')
            print(Xtr.shape, len(ytr))
            print(Xte.shape)
            return Xtr, ytr, Xte
        
        else:
            print('Calculating the mean to match oversampled data')
            vectors_per_author = dict() 
            auth_mean_vect_norm = dict()

            for vect, label in zip(Xtr, ytr):
                if label in vectors_per_author:
                    vectors_per_author[label].append(vect)
                else:
                    vectors_per_author[label] = [vect]

            for key in vectors_per_author.keys():
                mean_vect = np.mean(vectors_per_author[key], axis=0)
                norm_vect = mean_vect / np.linalg.norm(mean_vect)
                auth_mean_vect_norm[key] = norm_vect
            #return auth_mean_vect_norm

            start_idx = len(y_oversampled) - len(y_examples_generated) 
            new_vectors = [auth_mean_vect_norm[label] for label in y_oversampled[start_idx:]]
            Xtr = np.vstack([Xtr] + new_vectors)

            print(Xtr.shape, len(y_oversampled))
            print(Xte.shape)

            return Xtr, y_oversampled, Xte

            # for label in ytr[start_idx:]:
            #     Xtr = list(Xtr)
            #     Xtr.append(auth_mean_vect_norm[label])
            #     start_idx+=1


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
    def __init__(self, n=(1,4), use_idf=True, sublinear_tf=True, norm='l2', savecache='.postcache/dict.pkl', **tfidf_kwargs):
        #assert language in {'latin', 'spanish'}, 'the requested language is not yet covered'
        # if language == 'latin':
        #     language = 'lat'
        # self.language = language
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.tfidf_kwargs = tfidf_kwargs
        self.savecache = savecache
        self.n = n
        self.counter = CountVectorizer(analyzer=self.post_analyzer)
        #self.tagger=spacy.load('la_core_web_lg')
        # self.init_cache()
    
    def __str__(self) -> str:
        return f'FeaturesPOST [n-gram range: ({self.n[0]},{self.n[1]})]'


    def post_analyzer(self, doc):
        ngram_range = self.tfidf_kwargs.get('ngram_range', (self.n)) # up to quadrigrams
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


    def fit(self, documents, y=None):
        #self.tagger = POSTag(self.language) 
        #self.tagger = spacy.load('la_core_web_lg')
        self.count_pos_tags(documents)
        self.vectorizer = TfidfVectorizer(
            analyzer=self.post_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
        self.vectorizer.fit(documents)
        # self.save_cache()
        return self

    def transform(self, documents, y=None):
        self.count_pos_tags(documents)
        post_features = self.vectorizer.transform(documents)
        # self.save_cache()

        # features_num =post_features.shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return post_features

    def fit_transform(self, documents, y=None):
        #self.tagger = POSTag(self.language) # sostituire con spicy
        #self.tagger = spacy.load('la_core_web_lg')
        self.count_pos_tags(documents)
        self.vectorizer = TfidfVectorizer(
            analyzer=self.post_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
        post_features = self.vectorizer.fit_transform(documents)
        # self.save_cache()

        # features_num = post_features.shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return post_features

    def count_pos_tags(self, documents):
        if not hasattr(self, 'n_training_terms'):
            self.training_words = self.counter.fit_transform(documents)
            self.n_training_terms = self.training_words.sum(axis=1).getA().flatten()
        else:
            # Trasforma i nuovi testi e calcola il numero di n-grams
            self.test_words = self.counter.transform(documents)
            self.n_test_terms = self.test_words.sum(axis=1).getA().flatten()
    

class FeaturesDEP:
    def __init__(self, n=(1,3), use_idf=True, sublinear_tf=True, norm='l2', savecache='.depcache/dict.pkl', **tfidf_kwargs):
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.tfidf_kwargs = tfidf_kwargs
        self.savecache = savecache
        self.n = n
        self.counter = CountVectorizer(analyzer=self.dep_analyzer)
        # self.init_cache()
    
    def __str__(self) -> str:
        return f'FeaturesDEP [n-gram range: ({self.n[0]},{self.n[1]})]'
    

    def dep_analyzer(self, doc):
        ngram_range = self.tfidf_kwargs.get('ngram_range', (self.n))
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



    def fit(self, documents, y=None):
        #self.tagger = POSTag(self.language) # sostituire con spacy
        #self.tagger = spacy.load('la_core_web_lg')
        self.vectorizer = TfidfVectorizer(
            analyzer=self.dep_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
        self.vectorizer.fit(documents)
        #self.save_cache()
        return self

    def transform(self, documents, y=None):
        self.count_deps(documents)
        dep_features = self.vectorizer.transform(documents)
        #self.save_cache()

        features_num =dep_features.shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return dep_features

    def fit_transform(self, documents, y=None):
        #self.tagger = POSTag(self.language) # sostituire con spicy
        #self.tagger = spacy.load( 'la_core_web_lg')
        self.count_deps(documents)
        self.vectorizer = TfidfVectorizer(
            analyzer=self.dep_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
        dep_features = self.vectorizer.fit_transform(documents)
        #self.save_cache()

        # features_num =dep_features.shape[1]

        # print(f'Vectorizer: {self}')
        # print('Features:', features_num)
        return dep_features
    
    def count_deps(self, documents):
        if not hasattr(self, 'n_training_terms'):
            self.training_words = self.counter.fit_transform(documents)
            self.n_training_terms = self.training_words.sum(axis=1).getA().flatten()
        else:
            # Trasforma i nuovi testi e calcola il numero di n-grams
            self.test_words = self.counter.transform(documents)
            self.n_test_terms = self.test_words.sum(axis=1).getA().flatten()



# def oversample_DRO(self, Xtr, ytr, Xte, yte, n_trining_terms, n_test_terms):
#     is_sparse = issparse(Xtr)
#     if is_sparse:

#         print('Oversampling positive class using DRO method')

#         positives = ytr.sum()
#         nD = len(ytr) 

#         print('Before oversampling')
#         print(f'positives = {positives} (prevalence={positives*100/nD:.2f}%)')

#         Xtr, ytr = self.dro.fit_transform(Xtr, ytr, self.n_training_terms)
#         Xte = self.dro.transform(Xte, self.n_test_terms, samples=1)

#         positives = ytr.sum()
#         nD = len(ytr)
#         print('After oversampling')
#         print(f'positives = {positives} (prevalence={positives*100/nD:.2f}%)')
#         print(Xtr.shape, len(ytr))
#         print(Xte.shape, len(yte))

# def oversample_with_mean(Xtr, ytr, Xtr_, ytr_, start_idx):

#     if not issparse(Xtr):
#         # returns a dict (keys=label, values=mean_vect)
#         print('Calculating the mean to match oversampled data')
#         auth_mean_vect = dict() 
#         auth_mean_vect_norm = dict()

#         for vect, label in zip(Xtr, ytr):
#             if label in auth_mean_vect:
#                 auth_mean_vect[label].append(vect)
#             else:
#                 auth_mean_vect[label] = [vect]

#         for key in auth_mean_vect.keys():
#             mean_vect = np.mean(auth_mean_vect[key], axis=0)
#             norm_vect = mean_vect / np.linalg.norm(mean_vect)
#             auth_mean_vect_norm[key] =  norm_vect

#         for vect_, label_ in zip(Xtr_[start_idx:], ytr_[start_idx:]):
#             vect = hstack([vect_, auth_mean_vect_norm[label]])

#         #start_idx = Xtr.shape[0]
