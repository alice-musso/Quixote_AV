import pickle
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from authorship_verification import get_full_books, get_segments


@dataclass
class LoadedCorpora:
    train_corpus: list
    test_corpus: list


@dataclass
class ExperimentOutputs:
    score_path: Path
    results_path: Path
    ablation_path: Path
    decision_changes_path: Path
    score_table: object
    prediction_table: object
    ablation_table: object
    decision_change_table: object


@dataclass
class AuthorPredictionArtifacts:
    score_table: pd.DataFrame
    predicted_table: pd.DataFrame
    authors: list[str]
    scored_authors: list[str]


def zero_columns(X, column_indices):
    X_clean = X.copy()
    if not column_indices:
        return X_clean

    if sparse.issparse(X_clean):
        X_clean = X_clean.tocsc(copy=True)
        X_clean[:, column_indices] = 0
        X_clean.eliminate_zeros()
        return X_clean.tocsr()

    X_clean[:, column_indices] = 0
    return X_clean


class QuixoteInferenceExperiment:
    def __init__(self, config):
        self.config = config
        from results import ResultWriter

        self.result_writer = ResultWriter(config.results_inference)

    def _load_hyperparams_if_requested(self):
        if not self.config.load_hyperparams:
            return None

        hyper_path = Path(self.config.hyperparams_save)
        if not hyper_path.exists():
            raise FileNotFoundError(f"{hyper_path} does not exist")

        with hyper_path.open("rb") as hyper_file:
            return pickle.load(hyper_file)

    def _print_evaluation(self, label, evaluation):
        print(
            f"{label} books: "
            f"Acc={evaluation.books.accuracy * 100:.2f}% "
            f"F1={evaluation.books.f1 * 100:.2f}%"
        )
        print(
            f"{label} segments: "
            f"Acc={evaluation.segments.accuracy * 100:.2f}% "
            f"F1={evaluation.segments.f1 * 100:.2f}%"
        )

    def _display_tables(self, score_table, prediction_table, ablation_table, decision_change_table):
        print("\nScore table:")
        print(score_table.to_string(index=False))
        print("\nPrediction table:")
        print(prediction_table.to_string(index=False))
        print("\nAblation table:")
        print(ablation_table.to_string(index=False))
        print("\nDecision change table:")
        print(decision_change_table.to_string(index=False))

    def _load_corpora(self):
        from data_preparation.data_loader import binarize_corpus, load_corpus

        train_corpus = binarize_corpus(
            load_corpus(self.config.train_dir),
            positive_author=self.config.positive_author,
        )
        test_corpus = binarize_corpus(
            load_corpus(self.config.test_dir),
            positive_author=self.config.positive_author,
        )
        return LoadedCorpora(train_corpus=train_corpus, test_corpus=test_corpus)

    def _prepare_verifier(self, verifier, corpora):
        loaded_hyperparams = self._load_hyperparams_if_requested()
        return verifier.prepare_verifier(
            train_documents=corpora.train_corpus,
            test_documents=corpora.test_corpus,
            hyperparams=loaded_hyperparams,
            save_hyper_path=None if self.config.load_hyperparams else self.config.hyperparams_save,
        )

    def _evaluate_pre_ablation(self, verifier, verifier_artifacts):
        evaluation = verifier.evaluate_verifier(
            X=verifier_artifacts.X_train,
            y=verifier_artifacts.y,
            groups=verifier_artifacts.groups,
            hyperparams=verifier_artifacts.hyperparams,
        )
        self._print_evaluation("Pre-ablation", evaluation)
        return evaluation

    def _compute_ablation(self, verifier, verifier_artifacts, train_corpus):
        from quijote_classifier.quijote_experiment import QuijoteAblationExperiment
        from quijote_classifier.supervised_term_weighting.tsr_functions import (
            posneg_information_gain, pointwise_mutual_information
        )

        ablation_experiment = QuijoteAblationExperiment(
            target_title=self.config.target_title,
            positive_author=self.config.positive_author,
        )
        positive_author_books = ablation_experiment.cervantes_only(train_corpus)
        topic_documents, y_quijote, topic_groups = ablation_experiment.topic_labels(positive_author_books)
        positive_author_train_matrix = verifier.transform_documents_with_selection(
            topic_documents,
            verifier_artifacts.feature_selection,
        )
        feature_ranking, _ = ablation_experiment.compute_feature_ranking(
            X=positive_author_train_matrix,
            y=y_quijote,
            random_state=self.config.random_state,
            tsr_metric=posneg_information_gain,
        )

        classifier = verifier.new_classifier().set_params(
            C=verifier_artifacts.hyperparams["C"],
            class_weight=verifier_artifacts.hyperparams["class_weight"],
        )
        return ablation_experiment.ablate(
            feature_ranking=feature_ranking,
            X=positive_author_train_matrix,
            X_test=positive_author_train_matrix,
            y=y_quijote,
            groups=topic_groups,
            classifier=classifier,
            feature_names=verifier_artifacts.feature_selection.selected_feature_names,
        )

    def _apply_ablation_to_verifier_data(self, verifier_artifacts, ablation_artifacts):
        X_train_ablated = zero_columns(
            verifier_artifacts.X_train,
            ablation_artifacts.deleted_features,
        )
        X_test_ablated = zero_columns(
            verifier_artifacts.X_test,
            ablation_artifacts.deleted_features,
        )
        return X_train_ablated, X_test_ablated

    def _evaluate_post_ablation(self, verifier, verifier_artifacts, X_train_ablated):
        evaluation = verifier.evaluate_verifier(
            X=X_train_ablated,
            y=verifier_artifacts.y,
            groups=verifier_artifacts.groups,
            hyperparams=verifier_artifacts.hyperparams,
        )
        self._print_evaluation("Post-ablation", evaluation)
        return evaluation

    def _expanded_original_author_labels(self, books):
        labels = []
        groups = []
        for group_id, book in enumerate(books):
            labels.append(book.original_author)
            groups.append(group_id)
            for _ in book.segmented:
                labels.append(book.original_author)
                groups.append(group_id)
        return np.asarray(labels, dtype=object), groups

    def _new_author_estimator(self, verifier, verifier_artifacts):
        return verifier.new_classifier().set_params(
            C=verifier_artifacts.hyperparams["C"],
            class_weight=verifier_artifacts.hyperparams["class_weight"],
        )

    def _eligible_authors(self, books):
        author_counts = {}
        for book in books:
            author_counts[book.original_author] = author_counts.get(book.original_author, 0) + 1
        return sorted(author for author, count in author_counts.items() if count > 1)

    def _predictable_authors(self, books):
        return sorted({book.original_author for book in books})

    def _positive_class_scores(self, estimator, X):
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(X)
            classes = list(estimator.classes_)
            positive_index = classes.index(1)
            return np.asarray(probabilities[:, positive_index], dtype=float)
        if hasattr(estimator, "decision_function"):
            return np.asarray(estimator.decision_function(X), dtype=float).reshape(-1)
        return np.asarray(estimator.predict(X), dtype=float)

    def _predict_each_author(
        self,
        verifier,
        verifier_artifacts,
        X_train,
        X_test,
        train_author_labels,
        authors,
        score_authors=None,
    ):
        score_authors = set(authors if score_authors is None else score_authors)
        score_columns = {}
        predicted_columns = {}
        for author in authors:
            y_train = (np.asarray(train_author_labels) == author).astype(int)
            if np.unique(y_train).size < 2:
                continue
            estimator = self._new_author_estimator(verifier, verifier_artifacts)
            estimator.fit(X_train, y_train)
            predicted_columns[author] = estimator.predict(X_test)
            if author in score_authors:
                score_columns[author] = self._positive_class_scores(estimator, X_test)

        predicted_table = pd.DataFrame(predicted_columns)
        score_table = pd.DataFrame(score_columns)
        if predicted_table.empty:
            raise ValueError("No one-vs-rest author classifier could be trained.")
        return AuthorPredictionArtifacts(
            score_table=score_table,
            predicted_table=predicted_table,
            authors=list(predicted_table.columns),
            scored_authors=list(score_table.columns),
        )

    def _evaluate_each_author(self, verifier, verifier_artifacts, X, author_labels, groups, phase, authors):
        author_labels = np.asarray(author_labels, dtype=object)
        groups = np.asarray(groups)
        logo = LeaveOneGroupOut()
        rows = []

        for author in authors:
            y = (author_labels == author).astype(int)
            predictions = np.zeros_like(y)
            trainable_folds = 0
            untrainable_folds = 0

            for train_idx, test_idx in logo.split(X, y, groups):
                y_train = y[train_idx]
                if np.unique(y_train).size < 2:
                    predictions[test_idx] = 0
                    untrainable_folds += 1
                    continue

                estimator = clone(self._new_author_estimator(verifier, verifier_artifacts))
                estimator.fit(X[train_idx], y_train)
                predictions[test_idx] = estimator.predict(X[test_idx])
                trainable_folds += 1

            y_books, predictions_books = get_full_books(y, predictions, groups)
            y_segments, predictions_segments = get_segments(y, predictions, groups)
            rows.append(
                {
                    "phase": phase,
                    "author": author,
                    "scope": "books",
                    "accuracy": float(accuracy_score(y_books, predictions_books)),
                    "f1": float(f1_score(y_books, predictions_books, pos_label=1, zero_division=1.0)),
                    "trainable_folds": trainable_folds,
                    "untrainable_folds": untrainable_folds,
                }
            )
            rows.append(
                {
                    "phase": phase,
                    "author": author,
                    "scope": "segments",
                    "accuracy": float(accuracy_score(y_segments, predictions_segments)),
                    "f1": float(f1_score(y_segments, predictions_segments, pos_label=1, zero_division=1.0)),
                    "trainable_folds": trainable_folds,
                    "untrainable_folds": untrainable_folds,
                }
            )

        return pd.DataFrame(rows)

    def _track_decision_changes(
        self,
        verifier,
        verifier_artifacts,
        test_corpus,
        pre_ablation_predictions,
        train_author_labels,
        ablation_artifacts,
    ):
        if ablation_artifacts is None:
            return pd.DataFrame()

        tracked_author = self.config.positive_author
        if tracked_author not in pre_ablation_predictions.authors:
            print(f'\nDecision change tracing skipped: classifier "{tracked_author}" is not available.')
            return pd.DataFrame()

        rows = []
        for row_index, book in enumerate(test_corpus):
            pre_prediction = int(pre_ablation_predictions.predicted_table.iloc[row_index][tracked_author])
            rows.append(
                {
                    "title": book.title,
                    "author": book.original_author,
                    "classifier_author": tracked_author,
                    "pre_ablation_prediction": pre_prediction,
                    "decision_changed": False,
                    "change_at_deleted_order": None,
                    "change_feature_index": None,
                    "change_feature_name": None,
                    "post_change_prediction": pre_prediction,
                }
            )

        deleted_features = list(ablation_artifacts.deleted_features)
        deleted_feature_names = list(ablation_artifacts.deleted_feature_names)
        if not deleted_features:
            return pd.DataFrame(rows)

        X_train_current = verifier_artifacts.X_train.copy()
        X_test_current = verifier_artifacts.X_test.copy()
        remaining_changes = len(rows)

        for deleted_order, feature_index in enumerate(deleted_features, start=1):
            if remaining_changes == 0:
                print(
                    f"Decision tracing stopped early after {deleted_order - 1} deletions: "
                    f'all test books changed decision for "{tracked_author}".'
                )
                break
            X_train_current = zero_columns(X_train_current, [feature_index])
            X_test_current = zero_columns(X_test_current, [feature_index])
            predictions = self._predict_each_author(
                verifier,
                verifier_artifacts,
                X_train_current,
                X_test_current,
                train_author_labels=train_author_labels,
                authors=[tracked_author],
            )
            feature_name = deleted_feature_names[deleted_order - 1]

            for test_index, _book in enumerate(test_corpus):
                row = rows[test_index]
                if row["decision_changed"]:
                    continue
                label = int(predictions.predicted_table.iloc[test_index][tracked_author])
                if label != row["pre_ablation_prediction"]:
                    row["decision_changed"] = True
                    row["change_at_deleted_order"] = deleted_order
                    row["change_feature_index"] = feature_index
                    row["change_feature_name"] = feature_name
                    row["post_change_prediction"] = label
                    remaining_changes -= 1

        return pd.DataFrame(rows)

    def run(self):
        from authorship_verification import AuthorshipVerification

        corpora = self._load_corpora()
        verifier = AuthorshipVerification(self.config)
        verifier_artifacts = self._prepare_verifier(verifier, corpora)
        train_author_labels, train_author_groups = self._expanded_original_author_labels(corpora.train_corpus)
        predictable_authors = self._predictable_authors(corpora.train_corpus)
        eligible_authors = self._eligible_authors(corpora.train_corpus)

        pre_ablation_evaluation = self._evaluate_pre_ablation(
            verifier,
            verifier_artifacts,
        )

        if self.config.skip_ablation:
            print("Ablation skipped.")
            X_train_ablated = verifier_artifacts.X_train
            X_test_ablated = verifier_artifacts.X_test
            post_ablation_evaluation = pre_ablation_evaluation
            ablation_artifacts = None
        else:
            ablation_artifacts = self._compute_ablation(
                verifier,
                verifier_artifacts,
                corpora.train_corpus,
            )
            X_train_ablated, X_test_ablated = self._apply_ablation_to_verifier_data(
                verifier_artifacts,
                ablation_artifacts,
            )
            post_ablation_evaluation = self._evaluate_post_ablation(
                verifier,
                verifier_artifacts,
                X_train_ablated,
            )

        pre_ablation_author_scores = self._predict_each_author(
            verifier,
            verifier_artifacts,
            verifier_artifacts.X_train,
            verifier_artifacts.X_test,
            train_author_labels=train_author_labels,
            authors=predictable_authors,
            score_authors=eligible_authors,
        )

        post_ablation_author_scores = self._predict_each_author(
            verifier,
            verifier_artifacts,
            X_train_ablated,
            X_test_ablated,
            train_author_labels=train_author_labels,
            authors=predictable_authors,
            score_authors=eligible_authors,
        )

        author_score_table = pd.concat(
            [
                self._evaluate_each_author(
                    verifier,
                    verifier_artifacts,
                    verifier_artifacts.X_train,
                    train_author_labels,
                    train_author_groups,
                    phase="pre_ablation",
                    authors=eligible_authors,
                ),
                self._evaluate_each_author(
                    verifier,
                    verifier_artifacts,
                    X_train_ablated,
                    train_author_labels,
                    train_author_groups,
                    phase="post_ablation",
                    authors=eligible_authors,
                ),
            ],
            ignore_index=True,
        )

        from results import (
            ExperimentTables,
            build_ablation_table,
            build_decision_change_table,
            build_prediction_table,
            build_score_table,
        )

        tables = ExperimentTables(
            score_table=build_score_table(
                author_score_table=author_score_table,
                model_selection_score=verifier_artifacts.model_selection_score,
            ),
            prediction_table=build_prediction_table(
                pre_predictions=pre_ablation_author_scores,
                post_predictions=post_ablation_author_scores,
                test_corpus=corpora.test_corpus,
            ),
            ablation_table=build_ablation_table(ablation_artifacts) if ablation_artifacts is not None else pd.DataFrame(),
            decision_change_table=pd.DataFrame(),
        )
        self._display_tables(
            score_table=tables.score_table,
            prediction_table=tables.prediction_table,
            ablation_table=tables.ablation_table,
            decision_change_table=tables.decision_change_table,
        )

        if self.config.skip_decision_changes:
            print("\nDecision change tracing skipped.")
        else:
            print("\nTracing decision changes...")
            tables.decision_change_table = build_decision_change_table(
                self._track_decision_changes(
                    verifier=verifier,
                    verifier_artifacts=verifier_artifacts,
                    test_corpus=corpora.test_corpus,
                    pre_ablation_predictions=pre_ablation_author_scores,
                    train_author_labels=train_author_labels,
                    ablation_artifacts=ablation_artifacts,
                ).to_dict(orient="records")
            )
            print("\nDecision change table:")
            print(tables.decision_change_table.to_string(index=False))
        saved_results = self.result_writer.save_tables(tables)

        return ExperimentOutputs(
            score_path=saved_results.score_csv_path,
            results_path=saved_results.predictions_csv_path,
            ablation_path=saved_results.ablation_csv_path,
            decision_changes_path=saved_results.decision_changes_csv_path,
            score_table=tables.score_table,
            prediction_table=tables.prediction_table,
            ablation_table=tables.ablation_table,
            decision_change_table=tables.decision_change_table,
        )
