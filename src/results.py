from dataclasses import dataclass
from pathlib import Path
import unicodedata

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
    pre_posteriors = _posterior_table(pre_predictions.score_table)
    post_posteriors = _posterior_table(post_predictions.score_table)
    manuscript_columns = _manuscript_columns(test_corpus)
    rows = []
    for author in pre_predictions.authors:
        author_rows = [
            ("pre_ablation_prediction", pre_predictions.predicted_table),
            ("pre_ablation_score", pre_predictions.score_table),
            ("pre_ablation_posterior", pre_posteriors),
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
        "feature_index",
        "feature_name",
        "posneg_information_gain",
    ]
    ranking_positions = {
        feature_index: rank
        for rank, feature_index in enumerate(ablation_artifacts.feature_ranking, start=1)
    }
    rows = []
    for deleted_order, feature_index in enumerate(ablation_artifacts.deleted_features, start=1):
        rank = ranking_positions.get(feature_index)
        feature_name = ablation_artifacts.deleted_feature_names[deleted_order - 1]
        feature_score = ablation_artifacts.deleted_feature_scores[deleted_order - 1]
        rows.append(
            {
                "deleted_order": deleted_order,
                "rank": rank,
                "feature_index": feature_index,
                "feature_name": feature_name,
                "posneg_information_gain": feature_score,
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
