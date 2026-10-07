from dataclasses import dataclass, field
from pathlib import Path
import unicodedata

import pandas as pd


@dataclass
class ExperimentTables:
    score_table: pd.DataFrame
    prediction_table: pd.DataFrame
    ablation_table: pd.DataFrame
    decision_change_table: pd.DataFrame
    book_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    segment_report: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class SavedResults:
    score_csv_path: Path
    predictions_csv_path: Path
    ablation_csv_path: Path
    decision_changes_csv_path: Path
    score_json_path: Path
    predictions_json_path: Path
    ablation_json_path: Path
    decision_changes_json_path: Path
    book_report_csv_path: Path
    book_report_json_path: Path
    segment_report_csv_path: Path
    segment_report_json_path: Path


def build_performance_reports(books, target_author, evaluations):
    """Summarize held-out verifier predictions: full book first, then segments."""
    book_columns = [
        "title",
        "actual_author",
        "predicted_author_pre_ablation",
        "predicted_author_post_ablation",
    ]
    book_rows = [
        {"title": book.title, "actual_author": book.original_author}
        for book in books
    ]
    segment_columns = [
        "title",
        "actual_author",
        "total_segments",
        "segment_predicted_target_pre_ablation",
        "segment_predicted_not_target_pre_ablation",
        "segment_predicted_target_post_ablation",
        "segment_predicted_not_target_post_ablation",
    ]
    segment_rows = [
        {
            "title": book.title,
            "actual_author": book.original_author,
            "total_segments": len(book.segmented),
        }
        for book in books
    ]
    expected_rows = sum(1 + len(book.segmented) for book in books)
    for phase, evaluation in evaluations.items():
        predictions = evaluation.predictions
        if len(predictions) != expected_rows:
            raise ValueError("Predictions do not match the full-book and segment layout.")
        offset = 0
        for book_id, book in enumerate(books):
            total = len(book.segmented)
            predicted_target = bool(predictions[offset] == target_author)
            segment_predictions = predictions[offset + 1:offset + 1 + total]
            target_count = sum(label == target_author for label in segment_predictions)
            book_rows[book_id][f"predicted_author_{phase}"] = (
                target_author if predicted_target else f"Not{target_author}"
            )
            segment_rows[book_id][f"segment_predicted_target_{phase}"] = int(target_count)
            segment_rows[book_id][f"segment_predicted_not_target_{phase}"] = int(total - target_count)
            offset += 1 + total
    return pd.DataFrame(book_rows, columns=book_columns), pd.DataFrame(segment_rows, columns=segment_columns)


def build_score_table(author_score_table, model_selection_score):
    score_table = author_score_table.copy()
    score_table["model_selection_score"] = model_selection_score
    return score_table


def _posterior_table(score_table):
    return score_table.copy().astype(float)


def _normalized_title(title):
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", title.lower())
        if not unicodedata.combining(character)
    )


def _manuscript_sort_key(title):
    normalized_title = _normalized_title(title)
    part_order = [
        "prologo",
        "prima novelle",
        "nucleo",
        "seconda novelle",
        "quijote apocrifo",
    ]
    for order, part_name in enumerate(part_order):
        if part_name in normalized_title:
            return order, normalized_title
    return len(part_order), normalized_title


def _manuscript_columns(test_corpus):
    rows = []
    for row_index, book in enumerate(test_corpus):
        rows.append(
            {
                "row_index": row_index,
                "title": book.title,
                "sort_key": _manuscript_sort_key(book.title),
            }
        )
    return sorted(rows, key=lambda row: row["sort_key"])


def build_prediction_table(
    pre_predictions,
    post_predictions,
    test_corpus,
):
    """Export before/after predictions and posterior scores by manuscript."""
    pre_posteriors = _posterior_table(pre_predictions.score_table)
    post_posteriors = _posterior_table(post_predictions.score_table)
    manuscript_columns = _manuscript_columns(test_corpus)
    rows = []
    for author in pre_predictions.authors:
        author_rows = [
            ("pre_ablation_prediction", pre_predictions.predicted_table),
            ("pre_ablation_posterior", pre_posteriors),
            ("post_ablation_prediction", post_predictions.predicted_table),
            ("post_ablation_posterior", post_posteriors),
        ]
        for statistic, table in author_rows:
            row = {"author": author, "statistic": statistic}
            for column in manuscript_columns:
                row_index = column["row_index"]
                title = column["title"]
                if author in table.columns:
                    row[title] = table.iloc[row_index][author]
                else:
                    row[title] = pd.NA
            rows.append(row)
    return pd.DataFrame(rows)


def build_ablation_table(ablation_artifacts):
    columns = [
        "deleted_order",
        "rank",
        "combined_rank",
        "feature_index",
        "feature_name",
        "ig_cervantes_quijote_vs_cervantes_notquijote",
        "ig_cervantes_quijote_vs_everything_else",
    ]
    ranking_positions = {
        feature_index: rank
        for rank, feature_index in enumerate(ablation_artifacts.feature_ranking, start=1)
    }
    rows = []
    for deleted_order, feature_index in enumerate(ablation_artifacts.deleted_features, start=1):
        rank = ranking_positions.get(feature_index)
        feature_name = ablation_artifacts.deleted_feature_names[deleted_order - 1]
        cervantes_only_score = ablation_artifacts.deleted_feature_cervantes_only_scores[deleted_order - 1]
        everything_else_score = ablation_artifacts.deleted_feature_everything_else_scores[deleted_order - 1]
        combined_rank = ablation_artifacts.deleted_feature_combined_ranks[deleted_order - 1]
        rows.append(
            {
                "deleted_order": deleted_order,
                "rank": rank,
                "combined_rank": combined_rank,
                "feature_index": feature_index,
                "feature_name": feature_name,
                "ig_cervantes_quijote_vs_cervantes_notquijote": cervantes_only_score,
                "ig_cervantes_quijote_vs_everything_else": everything_else_score,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_decision_change_table(decision_change_rows):
    return pd.DataFrame(decision_change_rows)


class ResultWriter:
    def __init__(self, results_path: str):
        self.predictions_json_path = Path(results_path)
        self.predictions_csv_path = self.predictions_json_path.with_suffix(".csv")
        self.score_json_path = self.predictions_json_path.parent / "score.json"
        self.score_csv_path = self.predictions_json_path.parent / "score.csv"
        self.ablation_json_path = self.predictions_json_path.parent / "ablation.json"
        self.ablation_csv_path = self.predictions_json_path.parent / "ablation.csv"
        self.decision_changes_json_path = self.predictions_json_path.parent / "decision_changes.json"
        self.decision_changes_csv_path = self.predictions_json_path.parent / "decision_changes.csv"

        stem = self.predictions_json_path.stem
        directory = self.predictions_json_path.parent
        self.book_report_csv_path = directory / f"{stem}_book_report.csv"
        self.book_report_json_path = directory / f"{stem}_book_report.json"
        self.segment_report_csv_path = directory / f"{stem}_segment_report.csv"
        self.segment_report_json_path = directory / f"{stem}_segment_report.json"

    def save_tables(self, tables: ExperimentTables):
        tables.score_table.to_csv(self.score_csv_path, index=False)
        tables.prediction_table.to_csv(self.predictions_csv_path, index=False)
        tables.ablation_table.to_csv(self.ablation_csv_path, index=False)
        tables.decision_change_table.to_csv(self.decision_changes_csv_path, index=False)

        tables.score_table.to_json(self.score_json_path, orient="records", indent=4)
        tables.prediction_table.to_json(self.predictions_json_path, orient="records", indent=4)
        tables.ablation_table.to_json(self.ablation_json_path, orient="records", indent=4)
        tables.decision_change_table.to_json(self.decision_changes_json_path, orient="records", indent=4)

        tables.book_report.to_csv(self.book_report_csv_path, index=False)
        tables.book_report.to_json(self.book_report_json_path, orient="records", indent=4)
        tables.segment_report.to_csv(self.segment_report_csv_path, index=False)
        tables.segment_report.to_json(self.segment_report_json_path, orient="records", indent=4)

        return SavedResults(
            book_report_csv_path=self.book_report_csv_path,
            book_report_json_path=self.book_report_json_path,
            segment_report_csv_path=self.segment_report_csv_path,
            segment_report_json_path=self.segment_report_json_path,
            score_csv_path=self.score_csv_path,
            predictions_csv_path=self.predictions_csv_path,
            ablation_csv_path=self.ablation_csv_path,
            decision_changes_csv_path=self.decision_changes_csv_path,
            score_json_path=self.score_json_path,
            predictions_json_path=self.predictions_json_path,
            ablation_json_path=self.ablation_json_path,
            decision_changes_json_path=self.decision_changes_json_path,
        )
