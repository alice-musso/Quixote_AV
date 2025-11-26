from sklearn.base import BaseEstimator, clone, ClassifierMixin
import numpy as np
from sklearn.linear_model import LogisticRegression
import scipy


class ClassifierRange(ClassifierMixin, BaseEstimator):

    def __init__(self,
                 base_cls: BaseEstimator,
                 C=1,
                 class_weight=None,
                 feat_funct_words=None,
                 feat_post=None,
                 feat_mendenhall=None,
                 feat_dvex=None,
                 feat_char=None,
                 feat_punct=None,
                 feat_dep=None,
                 feat_sentlength=None,
                 feat_k_freq_words=None
    ):
        self.base_cls = base_cls
        self.feat_funct_words = feat_funct_words
        self.feat_post = feat_post
        self.feat_mendenhall = feat_mendenhall
        self.feat_dvex = feat_dvex
        self.feat_char = feat_char
        self.feat_punct = feat_punct
        self.feat_dep= feat_dep
        self.feat_sentlength = feat_sentlength
        self.feat_k_freq_words = feat_k_freq_words
        self.C = C
        self.class_weight = class_weight

    def _extract_all(self, X):
        def _hstack(blocks):
            if all(isinstance(b, np.ndarray) for b in blocks):
                funct = np.hstack
            else:
                funct = scipy.sparse.hstack
            return funct(blocks)

        feat_slices = [val for param, val in self.__dict__.items() if param.startswith('feat_') if val is not None]
        X_sel = _hstack([X[:,feat_slice] for feat_slice in feat_slices])
        return X_sel

    def fit(self, X, y):
        X = self._extract_all(X)
        self.base_cls = clone(self.base_cls)
        self.set_params(C=self.C, class_weight=self.class_weight)
        self.base_cls.fit(X, y)
        return self

    def predict(self, X):
        X = self._extract_all(X)
        return self.base_cls.predict(X)

    def predict_proba(self, X):
        X = self._extract_all(X)
        return self.base_cls.predict_proba(X)

    @property
    def classes_(self):
        return self.base_cls.classes_


