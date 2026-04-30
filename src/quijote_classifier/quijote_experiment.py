import warnings
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from data_preparation.data_loader import Book
from quijote_classifier.supervised_term_weighting.tsr_functions import (
    get_supervised_matrix,
    get_tsr_matrix,
)
from scipy.stats import binom

warnings.filterwarnings("ignore")


@dataclass
class TopicAblationArtifacts:
    feature_ranking: list[int]
    ranked_feature_names: list[str]
    deleted_features: list[int]
    deleted_feature_names: list[str]


@dataclass
class TopicFeatureRankingArtifacts:
    feature_ranking: list[int]
    tsr_matrix: np.ndarray
    X_train: object
    X_test: object
    y_train: np.ndarray
    y_test: np.ndarray


class QuijoteAblationExperiment:
    def __init__(self, target_title="Quijote", positive_author="Cervantes"):
        self.target_title = target_title
        self.positive_author = positive_author

    def cervantes_only(self, books: list[Book]):
        return [book for book in books if book.author == self.positive_author]

    def topic_labels(self, books: list[Book]):
        documents = []
        labels = []
        groups = []

        for group_id, book in enumerate(books):
            label = int(self.target_title.lower() in book.title.lower())
            documents.append(book.processed)
            labels.append(label)
            groups.append(group_id)
            if book.segmented is not None:
                for fragment in book.segmented:
                    documents.append(fragment)
                    labels.append(label)
                    groups.append(group_id)

        return documents, np.asarray(labels), groups

    def compute_feature_ranking(self, X, y, random_state=0, tsr_metric=None):
        class_counts = np.bincount(y)
        stratify = y if np.unique(y).size > 1 and min(class_counts) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=random_state,
            stratify=stratify,
        )
        label_matrix = np.asarray(y_train).reshape(-1, 1)
        supervised_matrix = get_supervised_matrix(X_train, label_matrix, n_jobs=-1)
        tsr_matrix = get_tsr_matrix(supervised_matrix, tsr_metric, n_jobs=-1).flatten()
        feature_ranking = np.argsort(tsr_matrix)[::-1]
        feature_ranking = [index for index in feature_ranking if tsr_matrix[index] > 0]
        return TopicFeatureRankingArtifacts(
            feature_ranking=feature_ranking,
            tsr_matrix=tsr_matrix,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

    def ablate(
        self,
        feature_ranking,
        X_train,
        X_test,
        y_train,
        y_test,
        classifier: BaseEstimator,
        feature_names=None,
    ):
        if np.unique(y_train).size < 2:
            raise ValueError("Ablation training split must contain both topic classes.")

        features_remaining = X_train.shape[1]
        remove_per_step = 10
        delete_pointer = 0
        deleted_features = []
        feature_names = list(feature_names or [])
        ranked_feature_names = [
            feature_names[index] if index < len(feature_names) else f"feature_{index}"
            for index in feature_ranking
        ]
        deleted_feature_names = []

        print(f'prevalence Quijote"s: {np.mean(y_train) * 100:.3f}%')
        X_train = X_train.copy()
        X_test = X_test.copy()

        has_candidates = True
        degenerated = False
        while has_candidates and not degenerated:
            estimator = clone(classifier)
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=1.0)
            print(
                f"Held-out split: Acc={acc * 100:.2f}% "
                f"F1={f1 * 100:.2f}% num-feats={features_remaining}"
            )

            positive_predictions = y_pred[y_test == 1]
            acc = np.mean(positive_predictions) if len(positive_predictions) else 0.0  # aka recall
            print(
                f"Held-out split: Recall={acc * 100:.2f}% "
            )

            def threshold_accuracy(n, alpha=0.01, p0=0.5):
                # k* = smallest k such that P(K >= k) <= alpha
                k_star = binom.isf(alpha, n, p0)  # inverse survival function
                return int(k_star), k_star / n

            #_, acc_threshold = threshold_accuracy(n=len(y))
            positive_count = int(np.sum(y_test))
            acc_threshold = 0.0
            if positive_count:
                _, acc_threshold = threshold_accuracy(n=positive_count)
            print(f"{acc_threshold=} (recall)")
            if acc <= acc_threshold:
                degenerated = True
                print("stop: classifier has degenerated")
            elif delete_pointer < len(feature_ranking):
                to_delete = feature_ranking[delete_pointer:delete_pointer + remove_per_step]
                X_train = self._zero_columns(X_train, to_delete)
                X_test = self._zero_columns(X_test, to_delete)
                deleted_features.extend(to_delete)
                deleted_feature_names.extend(
                    feature_names[index] if index < len(feature_names) else f"feature_{index}"
                    for index in to_delete
                )
                delete_pointer += remove_per_step
                features_remaining -= remove_per_step
                print("deleting candidates")
            else:
                has_candidates = False
                print("stop: no more candidates to remove")

        print(f"X ablated has shape {X_train.shape}")
        return TopicAblationArtifacts(
            feature_ranking=feature_ranking,
            ranked_feature_names=ranked_feature_names,
            deleted_features=deleted_features,
            deleted_feature_names=deleted_feature_names,
        )

    def _zero_columns(self, X, column_indices):
        if sparse.issparse(X):
            X = X.tocsc(copy=True)
            X[:, column_indices] = 0
            X.eliminate_zeros()
            return X.tocsr()

        X = X.copy()
        X[:, column_indices] = 0
        return X
