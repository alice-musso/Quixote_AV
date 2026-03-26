from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack, csr_matrix, issparse
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import Normalizer
import numpy as np
from tqdm import tqdm
from nltk import ngrams

from string import punctuation
sp_punctuation = punctuation + '¡¿'
from abc import ABC, abstractmethod


class FeatureExtractorAA(ABC):
    """
    Abstract class of all feature extractors for authorship analysis
    """
    @abstractmethod
    def num_dimensions(self):
        ...

    @abstractmethod
    def get_feature_names_out(self):
        ...


class CountTrackingMixin:
    def _update_term_counts(self, counter, items, training_attr, test_attr):
        if not hasattr(self, 'n_training_terms'):
            training_matrix = counter.fit_transform(items)
            setattr(self, training_attr, training_matrix)
            self.n_training_terms = training_matrix.sum(axis=1).getA().flatten()
        else:
            test_matrix = counter.transform(items)
            setattr(self, test_attr, test_matrix)
            self.n_test_terms = test_matrix.sum(axis=1).getA().flatten()


class VectorizerFeatureExtractor(FeatureExtractorAA, CountTrackingMixin):
    training_attr = 'training_terms'
    test_attr = 'test_terms'

    def _vectorizer_inputs(self, documents):
        return documents

    def _count_inputs(self, documents):
        return self._vectorizer_inputs(documents)

    def fit(self, documents, y=None):
        self.vectorizer.fit(self._vectorizer_inputs(documents))
        return self

    def transform(self, documents, y=None):
        self._update_term_counts(
            self.counter,
            self._count_inputs(documents),
            self.training_attr,
            self.test_attr,
        )
        return self.vectorizer.transform(self._vectorizer_inputs(documents))

    def fit_transform(self, documents, y=None):
        count_inputs = self._count_inputs(documents)
        self._update_term_counts(
            self.counter,
            count_inputs,
            self.training_attr,
            self.test_attr,
        )
        return self.vectorizer.fit_transform(self._vectorizer_inputs(documents))

    def num_dimensions(self):
        return len(self.vectorizer.vocabulary_)

    def get_feature_names_out(self):
        return np.asarray(self.vectorizer.get_feature_names_out(), dtype=object)


class FeaturesFrequentWords(VectorizerFeatureExtractor):

    def __init__(self, max_features=3000, ngram_range=(1, 1), analyzer='word', norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=True, remove_stopwords=None):

        self.vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, max_features=max_features,
                                          dtype=np.float64, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
                                          sublinear_tf=sublinear_tf, stop_words=remove_stopwords)
        self.counter = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, max_features=max_features,
                                       stop_words=remove_stopwords)

    def _vectorizer_inputs(self, documents):
        return [doc.text for doc in documents]


class FeaturesDistortedView(VectorizerFeatureExtractor):

    def __init__(self, function_words, method, ngram_range=(1,1), **tfidf_kwargs):
        assert method in {'DVEX', 'DVMA', 'DVSA'}, 'text distortion method not valid'
        self.function_words = function_words
        self.ngram_range = ngram_range
        self.tfidf_kwargs = tfidf_kwargs
        self.method = method
        self.counter = CountVectorizer()
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, **self.tfidf_kwargs)

    def __str__(self) -> str:
        ngram_range_str = f' [n-gram range: {self.ngram_range}]'
        if self.method=='DVEX':
            return 'FeaturesDVEX'+ ngram_range_str
        if self.method=='DVMA':
            return 'FeaturesDVMA'+ ngram_range_str
        if self.method=='DVSA':
            return 'FeaturesDVSA'+ ngram_range_str

    def _vectorizer_inputs(self, documents):
        return self.distortion(documents, method=self.method)
    
    def distortion(self, documents, method):
        if method == 'DVEX':
            dis_texts = self.dis_DVEX(documents)
        elif method =='DVMA':
            dis_texts = self.dis_DVMA(documents)
        elif method =='DVSA':
            dis_texts = self.dis_DVSA(documents)
        return dis_texts
    
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

    def dis_DVEX(self, documents):

        def DVEX(token):
            if len(token) <= 2:
                return token
            return token[0] + ('*' * (len(token) - 2)) + token[-1]

        dis_texts = []
        for doc in tqdm(documents, 'DV-EX distorting', total=len(documents)):
            tokens = [str(token) for token in doc]
            dis_text = [token if token in self.function_words else DVEX(token) for token in tokens]
            dis_texts.append(' '.join(dis_text))

        return dis_texts

class DummyTfidf:

    def __init__(self,upto, feature_type="word"):
        assert feature_type in {'word', 'sentence'}, 'feature type not valid'
        self.upto = upto
        self.prefix = f"{feature_type}_length" 

    def get_feature_names_out(self):
        return np.array([f"{self.prefix}_{i}" for i in range(1, self.upto)])


class FeaturesMendenhall(FeatureExtractorAA):
    def __init__(self,upto=25):
        self.upto = upto
        self.vectorizer = DummyTfidf(self.upto)

    def __str__(self) -> str:
        return 'FeaturesMendenhall'

    def fit(self, documents, y=None):
        return self

    def transform(self, documents, y=None):
        features = []
        for doc in tqdm(documents, 'Extracting word lengths', total=len(documents)):
            word_lengths = [len(str(token)) for token in doc]
            hist = np.histogram(word_lengths, bins=np.arange(1, self.upto), density=True)[0]
            distribution = np.cumsum(hist)
            features.append(distribution)
        return np.asarray(features)

    def fit_transform(self, documents, y=None):
        return self.fit(documents).transform(documents)

    def num_dimensions(self):
        return self.upto-2

    def get_feature_names_out(self):
        return self.vectorizer.get_feature_names_out()


class FeaturesSentenceLength(FeatureExtractorAA):
    def __init__(self, upto=1000):
        self.upto = upto

    def __str__(self) -> str:
        return 'FeaturesSentenceLength'

    def fit(self, documents, y=None):
        return self

    def transform(self, documents, y=None):
        features = []
        for doc in tqdm(documents, 'Extracting sentence lengths', total=len(documents)):
            sentence_lengths = []
            for sentence in doc.sents:
                sent_len = [len(str(token)) for token in sentence]
                sentence_lengths += sent_len
            hist = np.histogram(sentence_lengths, bins=np.arange(1, self.upto), density=True)[0]
            distribution = np.cumsum(hist)
            features.append(distribution)
        return np.asarray(features)

    def fit_transform(self, documents, y=None):
        return self.fit(documents).transform(documents)

    def num_dimensions(self):
        return self.upto-2

    def get_feature_names_out(self):
        return DummyTfidf(self.upto, feature_type="sentence").get_feature_names_out()


class FeaturesCharNGram(VectorizerFeatureExtractor):
    training_attr = 'training_ngrams'
    test_attr = 'test_ngrams'

    def __init__(self, n=(1,3), sublinear_tf=False, norm='l1'):
        self.n = n
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.counter = CountVectorizer(analyzer='char', ngram_range=self.n, min_df=3)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(self.n), use_idf=False, norm=self.norm, min_df=3)
    
    def __str__(self) -> str:
        return f'FeaturesCharNGram [n-gram range: ({self.n[0]},{self.n[1]})]'

    def _vectorizer_inputs(self, documents):
        return [doc.text for doc in documents]

    def fit_transform(self, documents, y=None):
        raw_documents = self._vectorizer_inputs(documents)
        self._update_term_counts(
            self.counter,
            raw_documents,
            self.training_attr,
            self.test_attr,
        )
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(self.n), use_idf=False, norm=self.norm, min_df=3)
        return self.vectorizer.fit_transform(raw_documents)


class FeaturesFunctionWords(VectorizerFeatureExtractor):
    training_attr = 'training_words'
    test_attr = 'test_words'

    def __init__(self, function_words, use_idf=False, sublinear_tf=False, norm='l1', ngram_range=(1,3)):
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.function_words = function_words
        self.ngram_range = ngram_range
        self.counter = CountVectorizer(vocabulary=self.function_words, min_df=1)
        self.vectorizer = TfidfVectorizer(
            vocabulary=self.function_words, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, ngram_range=self.ngram_range)
    
    def __str__(self) -> str:
        ngram_range_str = f' [n-gram range: {self.ngram_range}]'
        return 'FeaturesFunctionWords' + ngram_range_str

    def _vectorizer_inputs(self, documents):
        return [doc.text for doc in documents]

class FeaturesPunctuation(VectorizerFeatureExtractor):
    training_attr = 'training_words'
    test_attr = 'test_words'

    def __init__(self, sublinear_tf=False, norm='l1', ngram_range=(1,3)):
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.punctuation=sp_punctuation
        self.ngram_range = ngram_range
        self.counter = CountVectorizer(vocabulary=self.punctuation, min_df=1)
        self.vectorizer = TfidfVectorizer(analyzer='char', vocabulary=self.punctuation, use_idf=False, norm=self.norm, min_df=3, ngram_range=self.ngram_range)
    
    def __str__(self) -> str:
        ngram_range_str = f' [n-gram range: {self.ngram_range}]'
        return 'FeaturesPunctuation' + ngram_range_str

    def _vectorizer_inputs(self, documents):
        return [doc.text for doc in documents]


class FeaturesPOST(VectorizerFeatureExtractor):
    training_attr = 'training_words'
    test_attr = 'test_words'
    def __init__(self, n=(1,4), use_idf=True, sublinear_tf=True, norm='l2', savecache='.postcache/dict.pkl', **tfidf_kwargs):
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.tfidf_kwargs = tfidf_kwargs
        self.savecache = savecache
        self.n = n
        self.counter = CountVectorizer(analyzer=self.post_analyzer)
        self.vectorizer = TfidfVectorizer(analyzer=self.post_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
    
    def __str__(self) -> str:
        return f'FeaturesPOST [n-gram range: ({self.n[0]},{self.n[1]})]'

    def post_analyzer(self, doc):
        ngram_range = self.tfidf_kwargs.get('ngram_range', (self.n)) # up to quadrigrams
        ngram_range = slice(*ngram_range)
        ngram_tags = []
        
        for sentence in doc.sents:
            sentence_unigram_tags = [token.pos_ if token.pos_ != '' else 'Unk' for token in sentence]
            for n in list(range(ngram_range.start, ngram_range.stop+1)):
                sentence_ngram_tags = ['-'.join(ngram) for ngram in list(ngrams(sentence_unigram_tags, n))]
                ngram_tags.extend(sentence_ngram_tags)
        return ngram_tags

class FeaturesDEP(VectorizerFeatureExtractor):
    training_attr = 'training_words'
    test_attr = 'test_words'
    def __init__(self, n=(1,3), use_idf=True, sublinear_tf=True, norm='l2', savecache='.depcache/dict.pkl', use_words= False, **tfidf_kwargs):
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.tfidf_kwargs = tfidf_kwargs
        self.savecache = savecache
        self.n = n
        self.use_words = use_words
        self.counter = CountVectorizer(analyzer=self.dep_analyzer)
        self.vectorizer = TfidfVectorizer(analyzer=self.dep_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
    
    def __str__(self) -> str:
        return f'FeaturesDEP [n-gram range: ({self.n[0]},{self.n[1]})]'

    def dep_analyzer(self, doc):
        ngram_range = self.tfidf_kwargs.get('ngram_range', self.n)

        if self.use_words:
            word_dep_features = []
            for token in doc:
                if not token.is_punct and not token.is_space:
                    dep = token.dep_ if token.dep_ != '' else 'Unk'
                    word_dep_features.append(f"{token.text}:{dep}")
            return word_dep_features
        else:
            ngram_deps = []
            for sentence in doc.sents:
                sentence_features = []

                for token in sentence:
                    dep = token.dep_ if token.dep_ != '' else 'Unk'
                    sentence_features.append(dep)

                for n in range(ngram_range[0], ngram_range[1] + 1):
                    sentence_ngram_deps = ['-'.join(ngram) for ngram in list(ngrams(sentence_features, n))]
                    ngram_deps.extend(sentence_ngram_deps)

            return ngram_deps

class FeatureSetReductor(FeatureExtractorAA):

    def __init__(self, feature_extractor:FeatureExtractorAA, measure=chi2, max_features=5000, normalize=True, oversample=False):
        self.feature_extractor = feature_extractor
        self.max_features = max_features
        self.measure = measure
        self.normalize = normalize 
        self.oversample = oversample
        if (oversample==True):
            raise NotImplementedError('Oversample not yet implemented')
        self.is_sparse = True
        if self.normalize:
            self.normalizer = Normalizer()
        
    def __str__(self) -> str:
        return( f'FeatureSetReductor for {self.feature_extractor}' )

    def transform(self, X):
        matrix = self.feature_extractor.transform(X)
        matrix = self.feature_selector.transform(matrix)
        if self.normalize:
            matrix  = self.normalizer.transform(matrix)

        return matrix

    def fit_transform(self, X, y):
        matrix = self.feature_extractor.fit_transform(X, y)
        feature_dimensions = matrix.shape[1]

        self.feature_selector = SelectKBest(self.measure, k='all')
        if feature_dimensions > self.max_features:
            self.feature_selector = SelectKBest(self.measure, k=self.max_features)
            print(f'{self.feature_extractor}: reducing from {feature_dimensions} to {self.max_features}')
        else:
            self.normalize = False

        matrix = self.feature_selector.fit_transform(matrix, y)

        if self.normalize:
            matrix  = self.normalizer.fit_transform(matrix, y)

        return matrix

    def num_dimensions(self):
        return min(self.max_features, self.feature_extractor.num_dimensions())

    def get_feature_names_out(self):
        feature_names = np.asarray(self.feature_extractor.get_feature_names_out(), dtype=object)
        if not hasattr(self, "feature_selector"):
            return feature_names
        selected_indices = self.feature_selector.get_support(indices=True)
        return feature_names[selected_indices]


class HstackFeatureSet(FeatureExtractorAA):
    def __init__(self, *vectorizers, verbose=False):
        self.vectorizers = vectorizers
        self.verbose = verbose

    def transform(self, documents):
        feats = [v.transform(documents) for v in self.vectorizers]
        return self._hstack(feats)

    def fit_transform(self, documents, authors=None):
        feats = []
        for vectorizer in self.vectorizers:
            if self.verbose:
                print(f'extracting features with {vectorizer}')
            Xi = vectorizer.fit_transform(documents, authors)
            feats.append(Xi)
            if self.verbose:
                print(f'\textracted {Xi.shape[1]} features')
        X = self._hstack(feats)
        if self.verbose:
            print(f'final matrix has shape={X.shape}')
        return X

    def _hstack(self, feats):
        for i, f in enumerate(feats):
            if not issparse(f):
                if not (isinstance(f, np.ndarray) and f.dtype == np.float64): 
                    feats[i] = np.asarray(f).astype(np.float64)

        anysparse = any(map(issparse, feats))
        if anysparse:
            feats = [csr_matrix(f) for f in feats]
            feats = hstack(feats)
        else:
            feats = np.hstack(feats)
        return feats

    def num_dimensions(self):
        return sum([v.num_dimensions() for v in self.vectorizers])

    def get_feature_slices(self):
        from_ = 0
        slices = []
        for vect in self.vectorizers:
            to_ = from_ + vect.num_dimensions()
            s = slice(from_, to_)
            slices.append(s)
            from_ = to_
        return slices

    def get_feature_names_out(self):
        feature_names = [np.asarray(vectorizer.get_feature_names_out(), dtype=object) for vectorizer in self.vectorizers]
        if not feature_names:
            return np.asarray([], dtype=object)
        return np.concatenate(feature_names)
