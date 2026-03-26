import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import scipy
from scipy import sparse
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_predict, cross_val_score
from sklearn.svm import LinearSVC

from classifier_range import ClassifierRange
from data_preparation.data_loader import Book, get_spanish_function_words
from feature_extraction.features import (
    FeatureSetReductor,
    FeaturesCharNGram,
    FeaturesDEP,
    FeaturesDistortedView,
    FeaturesFrequentWords,
    FeaturesFunctionWords,
    FeaturesMendenhall,
    FeaturesPOST,
    FeaturesPunctuation,
    FeaturesSentenceLength,
    HstackFeatureSet,
)

warnings.filterwarnings("ignore")


def get_full_books(y, y_pred, groups):
    groups = np.asarray(groups)
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    seen = set()
    full_idx = []
    for i, group in enumerate(groups):
        if group not in seen:
            full_idx.append(i)
            seen.add(group)

    return y[full_idx], y_pred[full_idx]


def get_segments(y, y_pred, groups):
    groups = np.asarray(groups)
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)

    seen = set()
    segment_idx = []
    for i, group in enumerate(groups):
        if group in seen:
            segment_idx.append(i)
        else:
            seen.add(group)

    return y[segment_idx], y_pred[segment_idx]


@dataclass
class ClassificationMetrics:
    accuracy: float
    f1: float


@dataclass
class VerificationEvaluation:
    books: ClassificationMetrics
    segments: ClassificationMetrics


@dataclass
class VerificationRunArtifacts:
    X_train: object
    X_test: object | None
    selected_hyperparams: dict
    feature_selection: "FeatureSelection"
    y: np.ndarray
    groups: list[int]
    hyperparams: dict
    model_selection_score: float


@dataclass
class FeatureSelection:
    selected_feature_keys: list[str]
    original_hyperparams: dict
    normalized_hyperparams: dict
    selected_feature_names: list[str]

    def apply(self, X):
        blocks = [X[:, self.original_hyperparams[key]] for key in self.selected_feature_keys]
        stack = np.hstack if all(isinstance(block, np.ndarray) for block in blocks) else scipy.sparse.hstack
        return stack(blocks)


class AuthorshipVerification:
    """Prepare, select, and evaluate a binary authorship verifier."""

    def __init__(self, config):
        self.config = config
        self.best_params = None
        self.best_score = None
        self.hstacker = None

    def feature_extraction_fit(self, processed_docs, y):
        spanish_function_words = get_spanish_function_words()
        vectorizers_dict = {
            "feat_funct_words": FeaturesFunctionWords(
                function_words=spanish_function_words,
                ngram_range=(1, 1),
            ),
            "feat_post": FeatureSetReductor(
                FeaturesPOST(n=(1, 3)),
                max_features=self.config.max_features,
            ),
            "feat_mendenhall": FeaturesMendenhall(upto=27),
            "feat_sentlength": FeaturesSentenceLength(),
            "feat_dvex": FeatureSetReductor(
                FeaturesDistortedView(
                    method="DVEX",
                    function_words=spanish_function_words,
                ),
                max_features=self.config.max_features,
            ),
            "feat_punct": FeaturesPunctuation(),
            "feat_dep": FeatureSetReductor(
                FeaturesDEP(n=(2, 3), use_words=True),
                max_features=self.config.max_features,
            ),
            "feat_char": FeatureSetReductor(FeaturesCharNGram(n=(1, 3))),
            "feat_k_freq_words": FeaturesFrequentWords(),
        }

        feature_names, vectorizers = list(zip(*vectorizers_dict.items()))
        self.hstacker = HstackFeatureSet(*vectorizers, verbose=True)
        X = self.hstacker.fit_transform(processed_docs, y)
        slices = self.hstacker.get_feature_slices()
        slices_dict = {
            feature_name: feature_slice
            for feature_name, feature_slice in zip(feature_names, slices)
        }
        feature_names_dict = {
            feature_name: [
                f"{feature_name}:{feature_label}"
                for feature_label in vectorizer.get_feature_names_out()
            ]
            for feature_name, vectorizer in zip(feature_names, vectorizers)
        }
        return X, np.asarray(y), slices_dict, feature_names_dict

    def feature_extraction_transform(self, processed_docs):
        return self.hstacker.transform(processed_docs)

    def prepare_X_y(self, books: List[Book]):
        texts = []
        labels = []
        groups = []
        for group_id, book in enumerate(books):
            texts.append(book.processed)
            labels.append(book.author)
            groups.append(group_id)
            for segment in book.segmented:
                texts.append(segment)
                labels.append(book.author)
                groups.append(group_id)
        X, y, slices, feature_names = self.feature_extraction_fit(texts, labels)
        return X, y, slices, feature_names, groups

    def new_classifier(self):
        classifier_type = getattr(self.config, "classifier_type", "lr")
        print(f"Building classifier: {classifier_type}\n")

        if classifier_type == "lr":
            return LogisticRegression(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
        if classifier_type == "svm":
            return LinearSVC(random_state=self.config.random_state)
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    def prepare_range_classifier(self):
        return ClassifierRange(
            base_cls=self.new_classifier(),
            positive=self.config.positive_author,
        )

    def _resolve_feature_slices(self, hyperparams, slices):
        resolved_hyperparams = dict(hyperparams)
        for feature_name, feature_slice in list(resolved_hyperparams.items()):
            if feature_name.startswith("feat_") and feature_slice is not None:
                resolved_hyperparams[feature_name] = slices[feature_name]
        return resolved_hyperparams

    def _build_feature_selection(self, selected_params, feature_names_by_block):
        selected_feature_keys = [
            key
            for key, value in selected_params.items()
            if key.startswith("feat_") and value is not None
        ]
        if not selected_feature_keys:
            raise ValueError("No feature blocks selected.")

        normalized_slices = {}
        offset = 0
        for key in selected_feature_keys:
            original_slice = selected_params[key]
            width = original_slice.stop - original_slice.start
            normalized_slices[key] = slice(offset, offset + width)
            offset += width

        normalized_params = {
            key: value
            for key, value in selected_params.items()
            if not key.startswith("feat_")
        }
        normalized_params.update(normalized_slices)
        selected_feature_names = []
        for key in selected_feature_keys:
            selected_feature_names.extend(feature_names_by_block[key])
        return FeatureSelection(
            selected_feature_keys=selected_feature_keys,
            original_hyperparams=dict(selected_params),
            normalized_hyperparams=normalized_params,
            selected_feature_names=selected_feature_names,
        )

    def _save_hyperparams(self, hyperparams, save_hyper_path):
        if save_hyper_path is None:
            return
        os.makedirs(Path(save_hyper_path).parent, exist_ok=True)
        with Path(save_hyper_path).open("wb") as hyper_file:
            pickle.dump(
                hyperparams,
                hyper_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def transform_books_with_selection(self, books, feature_selection):
        return self.transform_documents_with_selection(
            [book.processed for book in books],
            feature_selection,
        )

    def transform_documents_with_selection(self, processed_docs, feature_selection):
        X = self.feature_extraction_transform(processed_docs)
        return feature_selection.apply(X)

    def prepare_verifier(
        self,
        train_documents,
        test_documents,
        hyperparams=None,
        save_hyper_path=None,
    ):
        X, y, slices, feature_names_by_block, groups = self.prepare_X_y(train_documents)
        X_test = self.feature_extraction_transform([book.processed for book in test_documents])

        if hyperparams is None:
            cls_range = self.prepare_range_classifier()
            selector = GridSearchCV(
                estimator=cls_range,
                param_grid={
                    "C": np.logspace(-4, 4, 9),
                    "class_weight": [None],
                    "feat_funct_words": [slices["feat_funct_words"], None],
                    "feat_post": [slices["feat_post"], None],
                    "feat_mendenhall": [slices["feat_mendenhall"], None],
                    "feat_sentlength": [slices["feat_sentlength"], None],
                    "feat_dvex": [slices["feat_dvex"], None],
                    "feat_punct": [slices["feat_punct"], None],
                    "feat_dep": [slices["feat_dep"], None],
                    "feat_char": [slices["feat_char"], None],
                    "feat_k_freq_words": [slices["feat_k_freq_words"], None],
                    "rebalance_ratio": [None],
                },
                cv=LeaveOneGroupOut(),
                refit=False,
                verbose=2,
                scoring=make_scorer(
                    f1_score,
                    pos_label=self.config.positive_author,
                    zero_division=1.0,
                ),
                n_jobs=-1,
            )
            selector.fit(X, y, groups=groups)
            resolved_hyperparams = selector.best_params_
            self.best_score = float(selector.best_score_)
            self._save_hyperparams(resolved_hyperparams, save_hyper_path)
        else:
            resolved_hyperparams = self._resolve_feature_slices(hyperparams, slices)
            cls_range = self.prepare_range_classifier()
            feature_selection = self._build_feature_selection(
                resolved_hyperparams,
                feature_names_by_block,
            )
            X_selected_for_scoring = feature_selection.apply(X)
            cv_scores = cross_val_score(
                estimator=clone(cls_range).set_params(**feature_selection.normalized_hyperparams),
                X=X_selected_for_scoring,
                y=y,
                groups=groups,
                cv=LeaveOneGroupOut(),
                scoring=make_scorer(
                    f1_score,
                    pos_label=self.config.positive_author,
                    zero_division=1.0,
                ),
                n_jobs=-1,
            )
            self.best_score = float(np.mean(cv_scores))

        self.best_params = resolved_hyperparams
        feature_selection = self._build_feature_selection(
            resolved_hyperparams,
            feature_names_by_block,
        )
        X_selected = feature_selection.apply(X)
        X_test_selected = feature_selection.apply(X_test)
        return VerificationRunArtifacts(
            X_train=X_selected,
            X_test=X_test_selected,
            selected_hyperparams=resolved_hyperparams,
            feature_selection=feature_selection,
            y=y,
            groups=groups,
            hyperparams=feature_selection.normalized_hyperparams,
            model_selection_score=self.best_score,
        )

    def evaluate_verifier(self, X, y, groups, hyperparams):
        predictions = cross_val_predict(
            estimator=self.prepare_range_classifier().set_params(**hyperparams),
            X=X,
            y=y,
            cv=LeaveOneGroupOut(),
            groups=groups,
            n_jobs=self.config.n_jobs,
        )

        y_books, predictions_books = get_full_books(y, predictions, groups)
        books = ClassificationMetrics(
            accuracy=float(accuracy_score(y_books, predictions_books)),
            f1=float(
                f1_score(
                    y_books,
                    predictions_books,
                    pos_label=self.config.positive_author,
                    zero_division=1.0,
                )
            ),
        )

        y_segments, predictions_segments = get_segments(y, predictions, groups)
        segments = ClassificationMetrics(
            accuracy=float(accuracy_score(y_segments, predictions_segments)),
            f1=float(
                f1_score(
                    y_segments,
                    predictions_segments,
                    pos_label=self.config.positive_author,
                    zero_division=1.0,
                )
            ),
        )
        return VerificationEvaluation(
            books=books,
            segments=segments,
        )
