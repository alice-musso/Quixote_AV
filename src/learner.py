from collections import Counter

from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, clone, ClassifierMixin
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import scipy

from oversampling.dro import DistributionalRandomOversampling as DRO, as_array_of_ints


class ClassifierRange(ClassifierMixin, BaseEstimator):

    def __init__(self,
                 base_cls: BaseEstimator,
                 positive: str,
                 negative: str=None,
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
                 feat_k_freq_words=None,
                 rebalance_ratio=None,
                 words_by_doc=1000,
                 test_samples=100,
                 calibrate=False
    ):

        self.base_cls = base_cls
        self.positive = positive
        self.negative = negative
        self.feat_funct_words = feat_funct_words  # sparse
        self.feat_post = feat_post                # sparse
        self.feat_mendenhall = feat_mendenhall    # dense
        self.feat_dvex = feat_dvex                # sparse
        self.feat_char = feat_char                # sparse
        self.feat_punct = feat_punct              # dense
        self.feat_dep = feat_dep                  # sparse
        self.feat_sentlength = feat_sentlength    # dense
        self.feat_k_freq_words = feat_k_freq_words # sparse
        self.rebalance_ratio = rebalance_ratio
        self.C = C
        self.class_weight = class_weight
        self.words_by_doc = words_by_doc
        self.test_samples = test_samples
        self.calibrate = calibrate

    def _extract_all(self, X, y=None):
        def _hstack(blocks):
            assert not all(b is None for b in blocks), 'all blocks are None'
            if len(blocks)==1:
                return blocks[0]
            func = np.hstack if all(isinstance(b, np.ndarray) for b in blocks) else scipy.sparse.hstack
            return func(blocks)

        doc_idx = np.arange(X.shape[0])
        sparse_feat_slices = [slice_i for slice_i in self.sparse_blocks if slice_i is not None]
        X_sparse = None
        if len(sparse_feat_slices)>0:
            X_sparse = _hstack([X[:, feat_slice] for feat_slice in sparse_feat_slices])

        dense_feat_slices  = [slice_i for slice_i in self.dense_blocks  if slice_i is not None]
        X_dense = None
        if len(dense_feat_slices)>0:
            X_dense  = _hstack([X[:, feat_slice] for feat_slice in dense_feat_slices])

        to_fit = y is not None
        if self.rebalance_ratio is not None and X_sparse is not None:
            if to_fit:
                self.dro = DRO(self.rebalance_ratio)
                X_sparse_dro, y = self.dro.fit_transform(X_sparse, y=y, words_by_doc=self.words_by_doc)
                samples = self.dro.samples
            else:
                if not hasattr(self, 'dro'):
                    raise ValueError('dro called before fit')
                X_sparse_dro = self.dro.transform(X_sparse, words_by_doc=self.words_by_doc, samples=self.test_samples)
                samples = self.test_samples

            if X_dense is not None:
                X_dense_dro = self.dro.oversampling_observed(X_dense, samples=samples)
                X_ = _hstack([X_dense_dro, X_sparse_dro])
            else:
                X_ = X_sparse_dro

            doc_idx = self.dro.oversampling_observed(doc_idx, samples=samples)
        else:
            X_ = _hstack([b for b in [X_dense, X_sparse] if b is not None])

        if to_fit:
            return X_, y, doc_idx
        else:
            return X_, doc_idx

    def fit(self, X, y):
        if self.negative is None:
            self.negative = f'Not{self.positive}'

        self.sparse_blocks = [self.feat_funct_words, self.feat_post, self.feat_dvex, self.feat_char, self.feat_dep, self.feat_k_freq_words]
        self.dense_blocks = [self.feat_mendenhall, self.feat_punct, self.feat_sentlength]

        y_bin = (np.asarray(y) == self.positive).astype(int)
        X, y_bin, idx = self._extract_all(X, y_bin)
        self.base_cls = clone(self.base_cls)
        self.set_params(C=self.C, class_weight=self.class_weight)

        if self.calibrate:
            self.calib = CalibratedClassifierCV(
                self.base_cls,
                cv=10,
                #method="isotonic",
                method="sigmoid",
                n_jobs=-1,
            ).fit(X, y_bin)
            self.base_cls = self.calib
        else:
            self.base_cls.fit(X, y_bin)
        return self

    def predict(self, X):
        X, idx = self._extract_all(X)
        y_bin = self.base_cls.predict(X)
        y_bin = majority_vote(y_bin, idx, hard=True)
        y_str = self._ybin2str(y_bin)
        return y_str

    def predict_proba(self, X):
        X, idx = self._extract_all(X)
        post = self.base_cls.predict_proba(X)
        post = majority_vote(post, idx, hard=False)
        return post

    def score(self, X, y, sample_weight=None):
        y_str_pred = self.predict(X)
        return (y_str_pred==y).mean()

    @property
    def classes_(self):
        return np.asarray([self.negative, self.positive])

    def _ybin2str(self, y_bin):
        y_str = np.asarray([self.positive if y_i==1 else self.negative for y_i in y_bin], dtype=str)
        return y_str


def majority_vote(y_hat, idx, hard=True):
    y_final = []
    for i in np.unique(idx):
        group = y_hat[idx == i]
        if hard:
            vote = int(np.mean(group) >= 0.5)
        else: # soft
            vote = np.mean(group, axis=0)
        y_final.append(vote)
    return np.asarray(y_final)

