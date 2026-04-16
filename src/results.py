from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ExperimentTables:
    score_table: pd.DataFrame
    prediction_table: pd.DataFrame
    ablation_table: pd.DataFrame
    decision_change_table: pd.DataFrame


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


def build_score_table(author_score_table, model_selection_score):
    score_table = author_score_table.copy()
    score_table["model_selection_score"] = model_selection_score
    return score_table


def _prediction_columns(predictions, prefix):
    columns = []
    for row_index in range(len(predictions.predicted_table)):
        row = {}
        for author in predictions.authors:
            row[f"{prefix}_pred_{author}"] = int(predictions.predicted_table.iloc[row_index][author])
            row[f"{prefix}_score_{author}"] = predictions.score_table.iloc[row_index][author]
        columns.append(row)
    return columns


def build_prediction_table(
    pre_predictions,
    post_predictions,
    test_corpus,
):
    pre_rows = _prediction_columns(
        predictions=pre_predictions,
        prefix="pre_ablation",
    )
    post_rows = _prediction_columns(
        predictions=post_predictions,
        prefix="post_ablation",
    )

    rows = []
    for row_index, book in enumerate(test_corpus):
        row = {
            "title": book.title,
            "author": book.original_author,
        }
        row.update(pre_rows[row_index])
        row.update(post_rows[row_index])
        rows.append(row)
    return pd.DataFrame(rows)


def build_ablation_table(ablation_artifacts):
    ranking_positions = {
        feature_index: rank
        for rank, feature_index in enumerate(ablation_artifacts.feature_ranking, start=1)
    }
    rows = []
    for deleted_order, feature_index in enumerate(ablation_artifacts.deleted_features, start=1):
        rank = ranking_positions.get(feature_index)
        feature_name = ablation_artifacts.deleted_feature_names[deleted_order - 1]
        rows.append(
            {
                "deleted_order": deleted_order,
                "rank": rank,
                "feature_index": feature_index,
                "feature_name": feature_name,
            }
        )
    return pd.DataFrame(rows)


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

    def save_tables(self, tables: ExperimentTables):
        tables.score_table.to_csv(self.score_csv_path, index=False)
        tables.prediction_table.to_csv(self.predictions_csv_path, index=False)
        tables.ablation_table.to_csv(self.ablation_csv_path, index=False)
        tables.decision_change_table.to_csv(self.decision_changes_csv_path, index=False)

        tables.score_table.to_json(self.score_json_path, orient="records", indent=4)
        tables.prediction_table.to_json(self.predictions_json_path, orient="records", indent=4)
        tables.ablation_table.to_json(self.ablation_json_path, orient="records", indent=4)
        tables.decision_change_table.to_json(self.decision_changes_json_path, orient="records", indent=4)

        return SavedResults(
            score_csv_path=self.score_csv_path,
            predictions_csv_path=self.predictions_csv_path,
            ablation_csv_path=self.ablation_csv_path,
            decision_changes_csv_path=self.decision_changes_csv_path,
            score_json_path=self.score_json_path,
            predictions_json_path=self.predictions_json_path,
            ablation_json_path=self.ablation_json_path,
            decision_changes_json_path=self.decision_changes_json_path,
        )
