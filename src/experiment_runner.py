import pickle
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from scipy import sparse
from scipy.stats import binom


@dataclass
class LoadedCorpora:
    train_corpus: list
    test_corpus: list


@dataclass
class ExperimentOutputs:
    score_path: Path
    results_path: Path
    ablation_path: Path
    score_table: object
    prediction_table: object
    ablation_table: object


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

    def _predict_test_labels(self, classifier, probabilities):
        class_labels = list(classifier.classes_)
        predicted_indices = probabilities.argmax(axis=1)
        return [class_labels[index] for index in predicted_indices]

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
            tsr_metric=pointwise_mutual_information,
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

    def _predict_test_probabilities(self, verifier, verifier_artifacts, X_train, X_test):
        from sklearn.calibration import CalibratedClassifierCV

        calibrated_classifier = CalibratedClassifierCV(
            verifier.new_classifier().set_params(
                C=verifier_artifacts.hyperparams["C"],
                class_weight=verifier_artifacts.hyperparams["class_weight"],
            ),
            cv=10,
            method="sigmoid",
            n_jobs=-1,
        ).fit(X_train, verifier_artifacts.y)
        probabilities = calibrated_classifier.predict_proba(X_test)
        return calibrated_classifier, probabilities

    def _track_decision_changes(
        self,
        verifier,
        verifier_artifacts,
        test_corpus,
        pre_ablation_classifier,
        pre_ablation_probabilities,
        ablation_artifacts,
    ):
        if ablation_artifacts is None:
            return pd.DataFrame()

        pre_labels = self._predict_test_labels(
            pre_ablation_classifier,
            pre_ablation_probabilities,
        )
        rows = [
            {
                "title": book.title,
                "author": book.author,
                "pre_ablation_predicted_author": pre_labels[row_index],
                "decision_changed": False,
                "change_at_deleted_order": None,
                "change_feature_index": None,
                "change_feature_name": None,
                "post_change_predicted_author": pre_labels[row_index],
            }
            for row_index, book in enumerate(test_corpus)
        ]

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
                    "all test books changed decision."
                )
                break
            X_train_current = zero_columns(X_train_current, [feature_index])
            X_test_current = zero_columns(X_test_current, [feature_index])
            classifier, probabilities = self._predict_test_probabilities(
                verifier,
                verifier_artifacts,
                X_train_current,
                X_test_current,
            )
            labels = self._predict_test_labels(classifier, probabilities)
            feature_name = deleted_feature_names[deleted_order - 1]

            for row_index, label in enumerate(labels):
                row = rows[row_index]
                if row["decision_changed"]:
                    continue
                if label != row["pre_ablation_predicted_author"]:
                    row["decision_changed"] = True
                    row["change_at_deleted_order"] = deleted_order
                    row["change_feature_index"] = feature_index
                    row["change_feature_name"] = feature_name
                    row["post_change_predicted_author"] = label
                    remaining_changes -= 1

        return pd.DataFrame(rows)

    def run(self):
        from authorship_verification import AuthorshipVerification

        corpora = self._load_corpora()
        verifier = AuthorshipVerification(self.config)
        verifier_artifacts = self._prepare_verifier(verifier, corpora)

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

        pre_ablation_classifier, pre_ablation_probabilities = self._predict_test_probabilities(
            verifier,
            verifier_artifacts,
            verifier_artifacts.X_train,
            verifier_artifacts.X_test,
        )

        post_ablation_classifier, post_ablation_probabilities = self._predict_test_probabilities(
            verifier,
            verifier_artifacts,
            X_train_ablated,
            X_test_ablated,
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
                pre_ablation_evaluation=pre_ablation_evaluation,
                post_ablation_evaluation=post_ablation_evaluation,
                model_selection_score=verifier_artifacts.model_selection_score,
            ),
            prediction_table=build_prediction_table(
                pre_classifier=pre_ablation_classifier,
                pre_probabilities=pre_ablation_probabilities,
                post_classifier=post_ablation_classifier,
                post_probabilities=post_ablation_probabilities,
                test_corpus=corpora.test_corpus,
                positive_author=self.config.positive_author,
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

        print("\nTracing decision changes...")
        tables.decision_change_table = build_decision_change_table(
            self._track_decision_changes(
                verifier=verifier,
                verifier_artifacts=verifier_artifacts,
                test_corpus=corpora.test_corpus,
                pre_ablation_classifier=pre_ablation_classifier,
                pre_ablation_probabilities=pre_ablation_probabilities,
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
            score_table=tables.score_table,
            prediction_table=tables.prediction_table,
            ablation_table=tables.ablation_table,
        )

def threshold_accuracy(n, alpha=0.05, p0=0.5):
    # k* = smallest k such that P(K >= k) <= alpha
    k_star = binom.isf(alpha, n, p0)  # inverse survival function
    return int(k_star), k_star / n
