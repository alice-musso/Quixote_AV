from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from .oversampling.dro import DistributionalRandomOversampling
from scipy.sparse import hstack, csr_matrix, issparse
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import Normalizer
import numpy as np
from tqdm import tqdm
from nltk import ngrams

from string import punctuation
sp_punctuation = punctuation + '¡¿'
from abc import ABC, abstractmethod

#from oversampling.dro import DistributionalRandomOversampling


# from dro import DistributionalRandomOversampling


class FeatureExtractorAA(ABC):
    """
    Abstract class of all feature extractors for authorship analysis
    """
    @abstractmethod
    def num_dimensions(self):
        ...


class FeaturesDistortedView(FeatureExtractorAA):

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
        return features

    def fit_transform(self, documents, y=None):
        distortions = self.distortion(documents, method=self.method)
        self.count_words(distortions)
        features = self.vectorizer.fit_transform(distortions)
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
            dis_texts.append(' '.join(dis_text))

        return dis_texts

    def num_dimensions(self):
        return len(self.vectorizer.vocabulary_)

    
        
class DummyTfidf:

    def __init__(self,upto, feature_type="word"):
        assert feature_type in {'word', 'sentence'}, 'feature type not valid'
        self.upto = upto
        self.prefix = f"{feature_type}_length" 

    def get_feature_names_out(self):
        return np.array([f"{self.prefix}_{i}" for i in range(1, self.upto)])


class FeaturesMendenhall(FeatureExtractorAA):
    """
    Extract features as the frequency of the words' lengths used in the documents,
    following the idea behind Mendenhall's Characteristic Curve of Composition
    """
    def __init__(self,upto=25):
        self.upto = upto
        self.vectorizer = DummyTfidf(self.upto)

    def __str__(self) -> str:
        return 'FeaturesMendenhall'

    def fit(self, documents, y=None):
        return self

    def transform(self, documents, y=None):
        features = []
        for doc in tqdm(documents, 'Extracting word lenghts', total=len(documents)):
            word_lengths = [len(str(token)) for token in doc]
            hist = np.histogram(word_lengths, bins=np.arange(1, self.upto), density=True)[0]
            distributuion = np.cumsum(hist)
            features.append(distributuion)
        return np.asarray(features)

    def fit_transform(self, documents, y=None):
        return self.fit(documents).transform(documents)

    def num_dimensions(self):
        return self.upto-2


class FeaturesSentenceLength(FeatureExtractorAA):
    def __init__(self, upto=1000):
        self.upto = upto

    def __str__(self) -> str:
        return 'FeaturesSentenceLength'

    def fit(self, documents, y=None):
        return self

    def transform(self, documents, y=None):
        features = []
        for doc in tqdm(documents, 'Extracting sentence lenghts', total=len(documents)):
            sentence_lengths = []
            for sentence in doc.sents:
                sent_len = [len(str(token)) for token in sentence]
                sentence_lengths += sent_len
            hist = np.histogram(sentence_lengths, bins=np.arange(1, self.upto), density=True)[0]
            distributuion = np.cumsum(hist)
            features.append(distributuion)
        return np.asarray(features)

    def fit_transform(self, documents, y=None):
        return self.fit(documents).transform(documents)

    def num_dimensions(self):
        return self.upto-2


class FeaturesCharNGram(FeatureExtractorAA):

    def __init__(self, n=(1,3), sublinear_tf=False, norm='l1'):
        self.n = n
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.counter = CountVectorizer(analyzer='char', ngram_range=self.n, min_df=3)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(self.n), use_idf=False, norm=self.norm, min_df=3)
    
    def __str__(self) -> str:
        return f'FeaturesCharNGram [n-gram range: ({self.n[0]},{self.n[1]})]'

    def fit(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.vectorizer.fit(raw_documents)
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
        if not hasattr(self, 'n_training_terms'):
            self.training_ngrams = self.counter.fit_transform(texts)
            self.n_training_terms = self.training_ngrams.sum(axis=1).getA().flatten()
        else:
            self.test_ngrams = self.counter.transform(texts)
            self.n_test_terms = self.test_ngrams.sum(axis=1).getA().flatten()

    def num_dimensions(self):
        return len(self.vectorizer.vocabulary_)


class FeaturesFunctionWords(FeatureExtractorAA):

    def __init__(self, function_words, use_idf=False, sublinear_tf=False, norm='l1', ngram_range=(1,3)):
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.function_words=function_words
        self.ngram_range=ngram_range
        self.counter = CountVectorizer(vocabulary=self.function_words, min_df=1)
        self.vectorizer = TfidfVectorizer(
            vocabulary=self.function_words, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, ngram_range=self.ngram_range)
    
    def __str__(self) -> str:
        ngram_range_str = f' [n-gram range: {self.ngram_range}]'
        return 'FeaturesFunctionWords' + ngram_range_str

    def fit(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.vectorizer.fit(raw_documents)
        return self

    def transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.count_words(raw_documents)
        return self.vectorizer.transform(raw_documents)

    def fit_transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.count_words(raw_documents)  
        features = self.vectorizer.fit_transform(raw_documents)
        return features
    
    def count_words(self, texts):
        if not hasattr(self, 'n_training_terms'):
            self.training_words = self.counter.fit_transform(texts)
            self.n_training_terms = self.training_words.sum(axis=1).getA().flatten()
        else:
            # Trasforma i nuovi testi e calcola il numero di n-grams
            self.test_words = self.counter.transform(texts)
            self.n_test_terms = self.test_words.sum(axis=1).getA().flatten()

    def num_dimensions(self):
        return len(self.vectorizer.vocabulary_)


class FeaturesVerbalEndings(FeatureExtractorAA):

    def __init__(self, verbal_endings, n=(1,1), extract_longest_match=False, use_idf=True, sublinear_tf=True, norm='l2', **tfidf_kwargs):
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.tfidf_kwargs = tfidf_kwargs
        self.n = n
        self.verbal_endings=verbal_endings
        self.extract_longest_match=extract_longest_match
        self.counter = CountVectorizer(analyzer=self.endings_analyzer, vocabulary=self.verbal_endings)
        self.vectorizer = TfidfVectorizer(analyzer=self.endings_analyzer, vocabulary=self.verbal_endings, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
        

    def __str__(self) -> str:
        return 'FeaturesVerbalEndings'
    

    def fit(self, documents, y=None):
        self.count_words(documents)   
        self.vectorizer.fit(documents)
        return self

    def transform(self, documents, y=None):
        self.count_words(documents)
        endings_features = self.vectorizer.transform(documents)
        return endings_features

    def fit_transform(self, documents, y=None):
        self.count_words(documents)
        endings_features = self.vectorizer.fit_transform(documents)
        return endings_features


    def endings_analyzer(self, doc):
        ngram_range = self.tfidf_kwargs.get('ngram_range', self.n) # up to quadrigrams
        ngram_range = slice(*ngram_range)
        doc_endings = []
        
        for sentence in doc.sents:
            sent_endings = []
            sentence_unigram_verbs = [token.text.lower() for token in sentence if token.pos_ == 'VERB']
            matching_endings = [ending for ending in self.verbal_endings if any(verb.endswith(ending) for verb in sentence_unigram_verbs)]
            if matching_endings:
                if self.extract_longest_match:
                    sent_endings.append(max(matching_endings, key=len))
                else:
                    sent_endings.extend(matching_endings)

            for n in list(range(ngram_range.start, ngram_range.stop+1)):
                sentence_ngram_endings = ['-'.join(ngram) for ngram in list(ngrams(sent_endings, n))]
                doc_endings.extend(sentence_ngram_endings)
        return doc_endings

    def count_words(self, documents):
        if not hasattr(self, 'n_training_terms'):
            self.training_words = self.counter.fit_transform(documents)
            self.n_training_terms = self.training_words.sum(axis=1).getA().flatten()
        else:
            self.test_words = self.counter.transform(documents)
            self.n_test_terms = self.test_words.sum(axis=1).getA().flatten()

    def num_dimensions(self):
        return len(self.vectorizer.vocabulary_)

        

class FeaturesPunctuation(FeatureExtractorAA):

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

    def fit(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.vectorizer.fit(raw_documents)
        return self

    def transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.count_words(raw_documents)
        return self.vectorizer.transform(raw_documents)

    def fit_transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.count_words(raw_documents)
        return self.vectorizer.fit_transform(raw_documents)

    def count_words(self, texts):
        if not hasattr(self, 'n_training_terms'):
            self.training_words = self.counter.fit_transform(texts)
            self.n_training_terms = self.training_words.sum(axis=1).getA().flatten()
        else:
            self.test_words = self.counter.transform(texts)
            self.n_test_terms = self.test_words.sum(axis=1).getA().flatten()

    def num_dimensions(self):
        return len(self.vectorizer.vocabulary_)


class FeaturesPOST(FeatureExtractorAA):
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

    def fit(self, documents, y=None):
        self.count_pos_tags(documents)
        self.vectorizer.fit(documents)
        return self

    def transform(self, documents, y=None):
        self.count_pos_tags(documents)
        post_features = self.vectorizer.transform(documents)
        return post_features

    def fit_transform(self, documents, y=None):
        self.count_pos_tags(documents)
        post_features = self.vectorizer.fit_transform(documents)
        return post_features

    def count_pos_tags(self, documents):
        if not hasattr(self, 'n_training_terms'):
            self.training_words = self.counter.fit_transform(documents)
            self.n_training_terms = self.training_words.sum(axis=1).getA().flatten()
        else:
            self.test_words = self.counter.transform(documents)
            self.n_test_terms = self.test_words.sum(axis=1).getA().flatten()

    def num_dimensions(self):
        return len(self.vectorizer.vocabulary_)


class FeaturesDEP(FeatureExtractorAA):
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

    def fit(self, documents, y=None):
        self.vectorizer.fit(documents)
        return self

    def transform(self, documents, y=None):
        self.count_deps(documents)
        dep_features = self.vectorizer.transform(documents)
        features_num =dep_features.shape[1]
        return dep_features

    def fit_transform(self, documents, y=None):
        self.count_deps(documents)
        dep_features = self.vectorizer.fit_transform(documents)

        return dep_features
    
    def count_deps(self, documents):
        if not hasattr(self, 'n_training_terms'):
            self.training_words = self.counter.fit_transform(documents)
            self.n_training_terms = self.training_words.sum(axis=1).getA().flatten()
        else:
            self.test_words = self.counter.transform(documents)
            self.n_test_terms = self.test_words.sum(axis=1).getA().flatten()

    def num_dimensions(self):
        return len(self.vectorizer.vocabulary_)


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
    
    def oversample_DRO(self, Xtr, ytr, Xte, yte, groups=None, rebalance_ratio=0.2, test_samples=100):
        raise NotImplementedError('oversample not yet implemented')

        if not isinstance(ytr, np.ndarray):
            ytr = np.array(ytr)
        self.dro = DistributionalRandomOversampling(rebalance_ratio=rebalance_ratio)
        samples = self.dro._samples_to_match_ratio(ytr)
        original_indices = self.dro.get_original_indices(Xtr, samples)
        y_oversampled = self.dro._oversampling_observed(ytr, samples)
        Xtr_old = Xtr.copy()

        if groups:
            groups = [group.split('_0')[0] for group in groups]
            groups_oversampled = []
            for group, i in zip(groups, samples):
                groups_oversampled.extend([group]*i)

        n_examples = samples.sum() - len(ytr)

        if hasattr(self.feature_extractor, 'n_training_terms'):
            print('Oversampling positive class using DRO method')
            self.n_training_terms =  self.feature_extractor.n_training_terms
            self.n_test_terms = self.feature_extractor.n_test_terms

            positives = ytr.sum()
            nD = len(ytr) 

            print('Before oversampling')
            print(f'positives = {positives} (prevalence={positives*100/nD:.2f}%)')

            Xtr, ytr = self.dro.fit_transform(Xtr, ytr, self.n_training_terms)
            Xte = self.dro.transform(Xte, self.n_test_terms, samples=test_samples) #new

            positives = ytr.sum()
            nD = len(ytr)
            print('After oversampling')
            print(f'positives = {positives} (prevalence={positives*100/nD:.2f}%)')
            print(Xtr.shape, len(ytr))
            print(Xte.shape)
        
        else:
            print('Duplicating vectors to match oversampled data')
            print('Type of Xtr and Xte', type(Xtr), type(Xte))

            Xtr = [Xtr[i] for i in original_indices]
            ytr = [ytr[i] for i in original_indices]

            Xtr = np.array(Xtr)
            Xte = np.array(Xte)
            
            if len(Xtr.shape) == 1:
                Xtr = Xtr.reshape(-1, 1)
            
             # Oversample Xte and yte to match test_samples
            Xte = np.tile(Xte, (test_samples, 1))  # Duplicate Xte to match test_samples
            yte = np.array([yte] * test_samples)  # Duplicate yte to match test_samples
            
           

            print(Xtr.shape, len(ytr))
            print(Xte.shape)

        return Xtr, ytr, Xte, yte, groups_oversampled

    def num_dimensions(self):
        return min(self.max_features, self.feature_extractor.num_dimensions())


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
