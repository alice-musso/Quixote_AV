import warnings
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from data_preparation.data_loader import Book
from scipy.stats import binom

warnings.filterwarnings("ignore")


@dataclass
class TopicAblationArtifacts:
    feature_ranking: list[int]
    ranked_feature_names: list[str]
    feature_scores: np.ndarray
    deleted_features: list[int]
    deleted_feature_names: list[str]
    deleted_feature_scores: list[float]


@dataclass
class TopicFeatureRankingArtifacts:
    feature_ranking: list[int]
    feature_scores: np.ndarray
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

    def corpus_labels(self, books: list[Book]):
        documents = []
        topic_labels = []
        author_labels = []
        groups = []

        for group_id, book in enumerate(books):
            topic_label = int(self.target_title.lower() in book.title.lower())
            author_label = int(book.author == self.positive_author)
            documents.append(book.processed)
            topic_labels.append(topic_label)
            author_labels.append(author_label)
            groups.append(group_id)
            if book.segmented is not None:
                for fragment in book.segmented:
                    documents.append(fragment)
                    topic_labels.append(topic_label)
                    author_labels.append(author_label)
                    groups.append(group_id)

        return (
            documents,
            np.asarray(topic_labels, dtype=int),
            np.asarray(author_labels, dtype=int),
            groups,
        )

    def compute_feature_ranking(self, X, topic_labels, author_labels, random_state=0):
        topic_labels = np.asarray(topic_labels, dtype=int)
        author_labels = np.asarray(author_labels, dtype=int)

        cervantes_quijote = (author_labels == 1) & (topic_labels == 1)
        cervantes_not_quijote = (author_labels == 1) & (topic_labels == 0)
        not_cervantes = author_labels == 0

        if not np.any(cervantes_quijote):
            raise ValueError("Missing Cervantes & Quijote instances for topic ablation.")
        if not np.any(cervantes_not_quijote):
            raise ValueError("Missing Cervantes & NotQuijote instances for topic ablation.")
        if not np.any(not_cervantes):
            raise ValueError("Missing NotCervantes background instances for topic ablation.")

        feature_scores = self._weighted_log_odds_with_background(
            X[cervantes_quijote],
            X[cervantes_not_quijote],
            X[not_cervantes],
        )
        feature_ranking = np.argsort(feature_scores)[::-1]
        feature_ranking = [index for index in feature_ranking if feature_scores[index] > 0]

        cervantes_mask = author_labels == 1
        X_cervantes = X[cervantes_mask]
        y_cervantes = topic_labels[cervantes_mask]
        class_counts = np.bincount(y_cervantes)
        stratify = y_cervantes if np.unique(y_cervantes).size > 1 and min(class_counts) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_cervantes,
            y_cervantes,
            test_size=0.3,
            random_state=random_state,
            stratify=stratify,
        )
        return TopicFeatureRankingArtifacts(
            feature_ranking=feature_ranking,
            feature_scores=feature_scores,
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
        feature_scores=None,
    ):
        if np.unique(y_train).size < 2:
            raise ValueError("Ablation training split must contain both topic classes.")

        features_remaining = X_train.shape[1]
        remove_per_step = 10
        delete_pointer = 0
        deleted_features = []
        feature_scores = np.asarray(feature_scores if feature_scores is not None else [])
        feature_names = list(feature_names or [])
        ranked_feature_names = [
            feature_names[index] if index < len(feature_names) else f"feature_{index}"
            for index in feature_ranking
        ]
        deleted_feature_names = []
        deleted_feature_scores = []

        print(f'prevalence Quijote"s: {np.mean(y_train) * 100:.3f}%')
        X_train = X_train.copy()
        X_test = X_test.copy()

        has_candidates = True
        degenerated = False

        def threshold_accuracy(n, alpha=0.01, p0=0.5):
            # k* = smallest k such that P(K >= k) <= alpha
            k_star = binom.isf(alpha, n, p0)  # inverse survival function
            return int(k_star), k_star / n

        _, acc_threshold = threshold_accuracy(n=len(y_test))

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
            recall = np.mean(positive_predictions) if len(positive_predictions) else 0.0
            print(
                f"Held-out split: Recall={recall * 100:.2f}% "
            )

            print(f"{acc_threshold=} (accuracy of a random classifier threshold)")
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
                deleted_feature_scores.extend(
                    float(feature_scores[index]) if index < len(feature_scores) else np.nan
                    for index in to_delete
                )
                print("last removed features:")
                for sequence_offset, feature_index in enumerate(to_delete, start=1):
                    rank = delete_pointer + sequence_offset
                    feature_name = (
                        feature_names[feature_index]
                        if feature_index < len(feature_names)
                        else f"feature_{feature_index}"
                    )
                    feature_score = (
                        float(feature_scores[feature_index])
                        if feature_index < len(feature_scores)
                        else np.nan
                    )
                    print(
                        f"  rank={rank:>4} index={feature_index:>6} "
                        f"score={feature_score:>9.4f} name={feature_name}"
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
            feature_scores=feature_scores,
            deleted_features=deleted_features,
            deleted_feature_names=deleted_feature_names,
            deleted_feature_scores=deleted_feature_scores,
        )

    def _weighted_log_odds_with_background(self, quijote_X, not_quijote_X, background_X, prior_floor=0.01):
        """Rank features with weighted log-odds and a strictly positive prior.

        The background corpus is used as an informative Dirichlet prior. Some
        selected features can be absent from that background, though; without a
        small floor, those zero-prior features can still lead to log(0).
        """
        quijote_counts = self._sum_feature_weights(quijote_X)
        not_quijote_counts = self._sum_feature_weights(not_quijote_X)
        background_counts = self._sum_feature_weights(background_X)

        prior = np.asarray(background_counts, dtype=float) + prior_floor
        if not np.any(prior > 0):
            prior = np.ones_like(prior, dtype=float) * prior_floor

        quijote_total = float(np.sum(quijote_counts))
        not_quijote_total = float(np.sum(not_quijote_counts))
        prior_total = float(np.sum(prior))

        quijote_posterior = quijote_counts + prior
        not_quijote_posterior = not_quijote_counts + prior

        quijote_other = (quijote_total + prior_total) - quijote_posterior
        not_quijote_other = (not_quijote_total + prior_total) - not_quijote_posterior

        epsilon = np.finfo(float).eps
        quijote_other = np.maximum(quijote_other, epsilon)
        not_quijote_other = np.maximum(not_quijote_other, epsilon)

        quijote_posterior = np.maximum(quijote_posterior, epsilon)
        not_quijote_posterior = np.maximum(not_quijote_posterior, epsilon)

        delta = np.log(quijote_posterior / quijote_other) - np.log(not_quijote_posterior / not_quijote_other)
        variance = (1.0 / quijote_posterior) + (1.0 / not_quijote_posterior)
        return delta / np.sqrt(variance)

    def _sum_feature_weights(self, X):
        if sparse.issparse(X):
            return np.asarray(X.sum(axis=0)).ravel().astype(float)
        return np.asarray(X, dtype=float).sum(axis=0)

    def _zero_columns(self, X, column_indices):
        if sparse.issparse(X):
            X = X.tocsc(copy=True)
            X[:, column_indices] = 0
            X.eliminate_zeros()
            return X.tocsr()

        X = X.copy()
        X[:, column_indices] = 0
        return X
