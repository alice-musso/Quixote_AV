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
    posneg_information_gain,
)
from scipy.stats import binom

warnings.filterwarnings("ignore")


@dataclass
class TopicAblationArtifacts:
    feature_ranking: list[int]
    ranked_feature_names: list[str]
    cervantes_only_scores: np.ndarray
    everything_else_scores: np.ndarray
    combined_ranks: np.ndarray
    deleted_features: list[int]
    deleted_feature_names: list[str]
    deleted_feature_cervantes_only_scores: list[float]
    deleted_feature_everything_else_scores: list[float]
    deleted_feature_combined_ranks: list[int]


@dataclass
class TopicFeatureRankingArtifacts:
    feature_ranking: list[int]
    cervantes_only_scores: np.ndarray
    everything_else_scores: np.ndarray
    combined_ranks: np.ndarray
    X_train: object
    X_test: object
    y_train: np.ndarray
    y_test: np.ndarray


class QuijoteAblationExperiment:
    def __init__(self, target_title="Quijote", positive_author="Cervantes", n_jobs=1):
        self.target_title = target_title
        self.positive_author = positive_author
        self.n_jobs = n_jobs

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

    def compute_feature_ranking(
        self,
        X,
        topic_labels,
        author_labels,
        random_state=0,
        ranking_mode="combined",
    ):
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

        cervantes_mask = author_labels == 1
        X_cervantes = X[cervantes_mask]
        y_cervantes = topic_labels[cervantes_mask]
        cervantes_only_scores = self._information_gain_scores(
            X_cervantes,
            y_cervantes,
        )

        y_everything_else = cervantes_quijote.astype(int)
        everything_else_scores = self._information_gain_scores(
            X,
            y_everything_else,
        )

        cervantes_only_ranks = self._rank_positions_desc(cervantes_only_scores)
        everything_else_ranks = self._rank_positions_desc(everything_else_scores)
        combined_ranks = np.maximum(cervantes_only_ranks, everything_else_ranks)
        tie_breaker = np.minimum(cervantes_only_ranks, everything_else_ranks)
        if ranking_mode == "combined":
            candidate_mask = (cervantes_only_scores > 0) & (everything_else_scores > 0)
            ranking_order = np.lexsort((tie_breaker, combined_ranks))
        elif ranking_mode == "cervantes_only":
            candidate_mask = cervantes_only_scores > 0
            ranking_order = np.argsort(cervantes_only_ranks, kind="stable")
        else:
            raise ValueError(f"Unsupported ranking_mode: {ranking_mode}")

        feature_ranking = [
            int(index)
            for index in ranking_order
            if candidate_mask[index]
        ]

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
            cervantes_only_scores=cervantes_only_scores,
            everything_else_scores=everything_else_scores,
            combined_ranks=combined_ranks,
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
        cervantes_only_scores=None,
        everything_else_scores=None,
        combined_ranks=None,
    ):
        if np.unique(y_train).size < 2:
            raise ValueError("Ablation training split must contain both topic classes.")

        features_remaining = X_train.shape[1]
        remove_per_step = 10
        delete_pointer = 0
        deleted_features = []
        cervantes_only_scores = np.asarray(cervantes_only_scores if cervantes_only_scores is not None else [])
        everything_else_scores = np.asarray(everything_else_scores if everything_else_scores is not None else [])
        combined_ranks = np.asarray(combined_ranks if combined_ranks is not None else [])
        feature_names = list(feature_names or [])
        ranked_feature_names = [
            feature_names[index] if index < len(feature_names) else f"feature_{index}"
            for index in feature_ranking
        ]
        deleted_feature_names = []
        deleted_feature_cervantes_only_scores = []
        deleted_feature_everything_else_scores = []
        deleted_feature_combined_ranks = []

        print(f'prevalence Quijote"s: {np.mean(y_train) * 100:.3f}%')
        print(f"positive candidate features available: {len(feature_ranking)}")
        X_train = X_train.copy()
        X_test = X_test.copy()

        has_candidates = True
        degenerated = False

        def threshold_accuracy(n, alpha=0.01, p0=0.5):
            # k* = smallest k such that P(K >= k) <= alpha
            k_star = binom.isf(alpha, n, p0)  # inverse survival function
            return int(k_star), k_star / n

        _, acc_threshold = threshold_accuracy(n=len(y_test))

        if not feature_ranking:
            print("stop: no positive candidates to remove")
            print(f"X ablated has shape {X_train.shape}")
            return TopicAblationArtifacts(
                feature_ranking=feature_ranking,
                ranked_feature_names=ranked_feature_names,
                cervantes_only_scores=cervantes_only_scores,
                everything_else_scores=everything_else_scores,
                combined_ranks=combined_ranks,
                deleted_features=deleted_features,
                deleted_feature_names=deleted_feature_names,
                deleted_feature_cervantes_only_scores=deleted_feature_cervantes_only_scores,
                deleted_feature_everything_else_scores=deleted_feature_everything_else_scores,
                deleted_feature_combined_ranks=deleted_feature_combined_ranks,
            )

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
                deleted_feature_cervantes_only_scores.extend(
                    float(cervantes_only_scores[index]) if index < len(cervantes_only_scores) else np.nan
                    for index in to_delete
                )
                deleted_feature_everything_else_scores.extend(
                    float(everything_else_scores[index]) if index < len(everything_else_scores) else np.nan
                    for index in to_delete
                )
                deleted_feature_combined_ranks.extend(
                    int(combined_ranks[index]) if index < len(combined_ranks) else -1
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
                    cervantes_only_score = (
                        float(cervantes_only_scores[feature_index])
                        if feature_index < len(cervantes_only_scores)
                        else np.nan
                    )
                    everything_else_score = (
                        float(everything_else_scores[feature_index])
                        if feature_index < len(everything_else_scores)
                        else np.nan
                    )
                    combined_rank = (
                        int(combined_ranks[feature_index])
                        if feature_index < len(combined_ranks)
                        else -1
                    )
                    print(
                        f"  rank={rank:>4} max_rank={combined_rank:>4} "
                        f"index={feature_index:>6} ig_cq_vs_cnq={cervantes_only_score:>9.4f} "
                        f"ig_cq_vs_all={everything_else_score:>9.4f} name={feature_name}"
                    )
                delete_pointer += remove_per_step
                features_remaining -= remove_per_step
                print("deleting candidates")
            else:
                has_candidates = False
                print("stop: no more positive candidates to remove")

        print(f"X ablated has shape {X_train.shape}")
        return TopicAblationArtifacts(
            feature_ranking=feature_ranking,
            ranked_feature_names=ranked_feature_names,
            cervantes_only_scores=cervantes_only_scores,
            everything_else_scores=everything_else_scores,
            combined_ranks=combined_ranks,
            deleted_features=deleted_features,
            deleted_feature_names=deleted_feature_names,
            deleted_feature_cervantes_only_scores=deleted_feature_cervantes_only_scores,
            deleted_feature_everything_else_scores=deleted_feature_everything_else_scores,
            deleted_feature_combined_ranks=deleted_feature_combined_ranks,
        )

    def _information_gain_scores(self, X, y):
        label_matrix = np.asarray(y, dtype=int).reshape(-1, 1)
        supervised_matrix = get_supervised_matrix(X, label_matrix, n_jobs=self.n_jobs)
        return get_tsr_matrix(supervised_matrix, posneg_information_gain, n_jobs=self.n_jobs).flatten()

    def _rank_positions_desc(self, scores):
        scores = np.asarray(scores, dtype=float)
        ranking = np.argsort(scores, kind="stable")[::-1]
        positions = np.empty_like(ranking)
        positions[ranking] = np.arange(1, len(scores) + 1)
        return positions

    def _zero_columns(self, X, column_indices):
        if sparse.issparse(X):
            X = X.tocsc(copy=True)
            X[:, column_indices] = 0
            X.eliminate_zeros()
            return X.tocsr()

        X = X.copy()
        X[:, column_indices] = 0
        return X
