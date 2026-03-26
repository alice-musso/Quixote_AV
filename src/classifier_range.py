from dataclasses import dataclass

import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV

from oversampling.dro import DistributionalRandomOversampling as DRO


def majority_vote(y_hat, idx, hard=True):
    y_final = []
    for group_id in np.unique(idx):
        group = y_hat[idx == group_id]
        vote = int(np.mean(group) >= 0.5) if hard else np.mean(group, axis=0)
        y_final.append(vote)
    return np.asarray(y_final)


@dataclass(frozen=True)
class FeatureBlockSpec:
    param_name: str
    is_sparse: bool


@dataclass
class SelectedFeatureBlocks:
    dense_slices: list
    sparse_slices: list

    @classmethod
    def from_estimator(cls, estimator):
        dense_slices = []
        sparse_slices = []
        for spec in estimator.FEATURE_SPECS:
            selected_slice = getattr(estimator, spec.param_name)
            if selected_slice is None:
                continue
            if spec.is_sparse:
                sparse_slices.append(selected_slice)
            else:
                dense_slices.append(selected_slice)
        return cls(dense_slices=dense_slices, sparse_slices=sparse_slices)

    @staticmethod
    def _stack_blocks(blocks):
        if not blocks:
            return None
        if len(blocks) == 1:
            return blocks[0]
        stack = np.hstack if all(isinstance(block, np.ndarray) for block in blocks) else scipy.sparse.hstack
        return stack(blocks)

    def _extract_family(self, X, feature_slices):
        if not feature_slices:
            return None
        return self._stack_blocks([X[:, feature_slice] for feature_slice in feature_slices])

    def extract(self, X):
        X_dense = self._extract_family(X, self.dense_slices)
        X_sparse = self._extract_family(X, self.sparse_slices)
        if X_dense is None and X_sparse is None:
            raise ValueError("No feature blocks selected.")
        return X_dense, X_sparse

    def combine(self, X_dense, X_sparse):
        return self._stack_blocks([block for block in (X_dense, X_sparse) if block is not None])


class ClassifierRange(ClassifierMixin, BaseEstimator):
    FEATURE_SPECS = (
        FeatureBlockSpec("feat_funct_words", is_sparse=True),
        FeatureBlockSpec("feat_post", is_sparse=True),
        FeatureBlockSpec("feat_dvex", is_sparse=True),
        FeatureBlockSpec("feat_char", is_sparse=True),
        FeatureBlockSpec("feat_dep", is_sparse=True),
        FeatureBlockSpec("feat_k_freq_words", is_sparse=True),
        FeatureBlockSpec("feat_mendenhall", is_sparse=False),
        FeatureBlockSpec("feat_punct", is_sparse=False),
        FeatureBlockSpec("feat_sentlength", is_sparse=False),
    )

    def __init__(
        self,
        base_cls: BaseEstimator,
        positive: str,
        negative: str = None,
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
        calibrate=False,
    ):
        self.base_cls = base_cls
        self.positive = positive
        self.negative = negative
        self.feat_funct_words = feat_funct_words
        self.feat_post = feat_post
        self.feat_mendenhall = feat_mendenhall
        self.feat_dvex = feat_dvex
        self.feat_char = feat_char
        self.feat_punct = feat_punct
        self.feat_dep = feat_dep
        self.feat_sentlength = feat_sentlength
        self.feat_k_freq_words = feat_k_freq_words
        self.rebalance_ratio = rebalance_ratio
        self.C = C
        self.class_weight = class_weight
        self.words_by_doc = words_by_doc
        self.test_samples = test_samples
        self.calibrate = calibrate

    def _selected_feature_blocks(self):
        return SelectedFeatureBlocks.from_estimator(self)

    def _apply_rebalancing(self, selected_blocks, X_dense, X_sparse, y=None):
        doc_idx = np.arange(X_sparse.shape[0] if X_sparse is not None else X_dense.shape[0])
        is_fit = y is not None

        if self.rebalance_ratio is None or X_sparse is None:
            return selected_blocks.combine(X_dense, X_sparse), y, doc_idx

        if is_fit:
            self.dro = DRO(self.rebalance_ratio)
            X_sparse_rebalanced, y = self.dro.fit_transform(
                X_sparse,
                y=y,
                words_by_doc=self.words_by_doc,
            )
            samples = self.dro.samples
        else:
            if not hasattr(self, "dro"):
                raise ValueError("DRO transform called before fit")
            X_sparse_rebalanced = self.dro.transform(
                X_sparse,
                words_by_doc=self.words_by_doc,
                samples=self.test_samples,
            )
            samples = self.test_samples

        X_dense_rebalanced = None
        if X_dense is not None:
            X_dense_rebalanced = self.dro.oversampling_observed(X_dense, samples=samples)

        doc_idx = self.dro.oversampling_observed(doc_idx, samples=samples)
        return selected_blocks.combine(X_dense_rebalanced, X_sparse_rebalanced), y, doc_idx

    def _extract_selected_features(self, X, y=None):
        selected_blocks = self._selected_feature_blocks()
        X_dense, X_sparse = selected_blocks.extract(X)
        return self._apply_rebalancing(selected_blocks, X_dense, X_sparse, y=y)

    def _build_estimator(self):
        estimator = clone(self.base_cls).set_params(C=self.C, class_weight=self.class_weight)
        if self.calibrate:
            return CalibratedClassifierCV(
                estimator,
                cv=10,
                method="sigmoid",
                n_jobs=-1,
            )
        return estimator

    def fit(self, X, y):
        if self.negative is None:
            self.negative = f"Not{self.positive}"

        y_bin = (np.asarray(y) == self.positive).astype(int)
        X_selected, y_bin, _ = self._extract_selected_features(X, y=y_bin)
        estimator = self._build_estimator()
        estimator.fit(X_selected, y_bin)
        self.base_cls = estimator
        return self

    def predict(self, X):
        X_selected, _, doc_idx = self._extract_selected_features(X)
        y_bin = self.base_cls.predict(X_selected)
        return self._ybin2str(majority_vote(y_bin, doc_idx, hard=True))

    def predict_proba(self, X):
        X_selected, _, doc_idx = self._extract_selected_features(X)
        posteriors = self.base_cls.predict_proba(X_selected)
        return majority_vote(posteriors, doc_idx, hard=False)

    @property
    def classes_(self):
        return np.array([self.negative, self.positive])

    def _ybin2str(self, y_bin):
        return np.asarray(
            [self.positive if label == 1 else self.negative for label in y_bin],
            dtype=str,
        )
