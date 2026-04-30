import argparse
from pathlib import Path
from types import SimpleNamespace
import pickle

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION_PATH = PROJECT_ROOT / "results" / "inference" / "ablation.csv"
DEFAULT_DECISION_CHANGES_PATH = (
    PROJECT_ROOT / "results" / "inference" / "decision_changes.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "inference"
DEFAULT_TRAIN_DIR = (
    PROJECT_ROOT / "corpus" / "training"
    if (PROJECT_ROOT / "corpus" / "training").exists()
    else PROJECT_ROOT / "corpus"
)
DEFAULT_HYPERPARAMS_PATH = PROJECT_ROOT / "hyperparams" / "hyperparameters.pkl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot posneg information gain for deleted ablation features in deletion order."
    )
    parser.add_argument("--ablation-path", default=str(DEFAULT_ABLATION_PATH))
    parser.add_argument(
        "--decision-changes-path",
        default=str(DEFAULT_DECISION_CHANGES_PATH),
        help=(
            "Optional decision_changes.csv/json path. Use an empty string to skip "
            "decision-change markers."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default="ablation_posneg_information_gain")
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--positive-author", default="Cervantes")
    parser.add_argument("--target-title", default="Quijote")
    parser.add_argument("--classifier-type", choices=["lr", "svm"], default="lr")
    parser.add_argument("--hyperparams-save", default=str(DEFAULT_HYPERPARAMS_PATH))
    parser.add_argument(
        "--load-hyperparams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --no-load-hyperparams to rerun model selection.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=5000)
    return parser.parse_args()


def load_ablation_table(path):
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def load_optional_table(path):
    if path is None or str(path).strip() == "":
        return pd.DataFrame()
    path = Path(path).expanduser()
    if not path.exists():
        return pd.DataFrame()
    if path.stat().st_size <= 1:
        return pd.DataFrame()
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def build_verifier_config(args):
    return SimpleNamespace(
        positive_author=args.positive_author,
        classifier_type=args.classifier_type,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        max_features=args.max_features,
    )


def resolve_hyperparams_path(args):
    base_path = Path(args.hyperparams_save).expanduser()
    return base_path.parent / f"hyperparameters_posauth_{args.positive_author}.pkl"


def load_hyperparams(path):
    with Path(path).open("rb") as hyper_file:
        return pickle.load(hyper_file)


def prepare_topic_matrix(args):
    from authorship_verification import AuthorshipVerification
    from data_preparation.data_loader import binarize_corpus, load_corpus
    from quijote_classifier.quijote_experiment import QuijoteAblationExperiment

    hyperparams_path = resolve_hyperparams_path(args)
    hyperparams = None
    if args.load_hyperparams and not hyperparams_path.exists():
        raise RuntimeError(
            f"{hyperparams_path} does not exist. Run inference once to save "
            "hyperparameters, or pass --no-load-hyperparams to rerun model selection."
        )
    if args.load_hyperparams:
        hyperparams = load_hyperparams(hyperparams_path)

    train_corpus = binarize_corpus(
        load_corpus(
            args.train_dir,
            cache_path=str(PROJECT_ROOT / "src" / "data_preparation" / ".cache"),
        ),
        positive_author=args.positive_author,
    )
    verifier = AuthorshipVerification(build_verifier_config(args))
    verifier_artifacts = verifier.prepare_verifier(
        train_documents=train_corpus,
        test_documents=train_corpus,
        hyperparams=hyperparams,
        save_hyper_path=None if args.load_hyperparams else hyperparams_path,
    )

    ablation_experiment = QuijoteAblationExperiment(
        target_title=args.target_title,
        positive_author=args.positive_author,
    )
    positive_author_books = ablation_experiment.cervantes_only(train_corpus)
    topic_documents, y_quijote, _topic_groups = ablation_experiment.topic_labels(
        positive_author_books
    )
    topic_matrix = verifier.transform_documents_with_selection(
        topic_documents,
        verifier_artifacts.feature_selection,
    )
    return topic_matrix, y_quijote


def compute_deleted_posneg_information_gain(topic_matrix, y_quijote, deleted_feature_indices):
    from quijote_classifier.supervised_term_weighting.tsr_functions import (
        get_supervised_matrix,
        get_tsr_matrix,
        posneg_information_gain,
    )

    deleted_matrix = topic_matrix[:, deleted_feature_indices]
    label_matrix = np.asarray(y_quijote).reshape(-1, 1)
    supervised_matrix = get_supervised_matrix(deleted_matrix, label_matrix, n_jobs=-1)
    return get_tsr_matrix(supervised_matrix, posneg_information_gain, n_jobs=-1).flatten()


def build_posneg_table(ablation_table, posneg_information_gain_values=None):
    table = ablation_table.copy()
    if posneg_information_gain_values is not None:
        table["posneg_information_gain"] = posneg_information_gain_values
    if "posneg_information_gain" not in table.columns:
        raise ValueError(
            'Ablation table must contain "posneg_information_gain", or values must be provided.'
        )
    if "deleted_order" in table.columns:
        table = table.sort_values("deleted_order").reset_index(drop=True)
    elif "rank" in table.columns:
        table = table.sort_values("rank").reset_index(drop=True)
    else:
        table = table.reset_index(drop=True)
    table["sequence_order"] = np.arange(1, len(table) + 1)
    table["cumulative_posneg_information_gain"] = table[
        "posneg_information_gain"
    ].cumsum()
    total_posneg_information_gain = table["posneg_information_gain"].sum()
    table["cumulative_posneg_information_gain_fraction"] = (
        table["cumulative_posneg_information_gain"] / total_posneg_information_gain
        if total_posneg_information_gain
        else 0.0
    )
    return table


def cumulative_threshold_row(posneg_table, threshold=0.9):
    if posneg_table.empty:
        return None
    if "cumulative_posneg_information_gain_fraction" not in posneg_table.columns:
        raise ValueError(
            'Posneg table must contain "cumulative_posneg_information_gain_fraction".'
        )

    threshold_rows = posneg_table[
        posneg_table["cumulative_posneg_information_gain_fraction"] >= threshold
    ]
    if threshold_rows.empty:
        return None
    return threshold_rows.iloc[0]


def elbow_row(posneg_table):
    if len(posneg_table) < 3:
        return None

    points = posneg_table[
        ["sequence_order", "posneg_information_gain"]
    ].to_numpy(dtype=float)
    start = points[0]
    end = points[-1]
    baseline = end - start
    baseline_norm = np.linalg.norm(baseline)
    if baseline_norm == 0.0:
        return None

    offsets = points - start
    distances = np.abs(
        baseline[0] * offsets[:, 1] - baseline[1] * offsets[:, 0]
    ) / baseline_norm
    elbow_index = int(np.argmax(distances))
    return posneg_table.iloc[elbow_index]


def decision_change_rows(decision_change_table, classifier_author="Cervantes"):
    if decision_change_table.empty:
        return pd.DataFrame()
    required_columns = {"change_at_deleted_order", "decision_changed"}
    if not required_columns.issubset(decision_change_table.columns):
        return pd.DataFrame()

    table = decision_change_table.copy()
    if "classifier_author" in table.columns:
        table = table[table["classifier_author"] == classifier_author]
    table = table[
        table["decision_changed"].astype(str).str.lower().isin(["true", "1", "yes"])
    ]
    table = table.dropna(subset=["change_at_deleted_order"])
    if table.empty:
        return table

    table["change_at_deleted_order"] = table["change_at_deleted_order"].astype(int)
    sort_columns = ["change_at_deleted_order"]
    if "title" in table.columns:
        sort_columns.append("title")
    return table.sort_values(sort_columns)


def decision_change_summary(posneg_table, decision_rows):
    if decision_rows.empty:
        return pd.DataFrame()

    summary = (
        decision_rows.groupby("change_at_deleted_order")
        .size()
        .rename("decision_change_count")
        .reset_index()
    )
    summary = summary.rename(columns={"change_at_deleted_order": "sequence_order"})
    metadata_columns = [
        column
        for column in [
            "sequence_order",
            "rank",
            "feature_index",
            "feature_name",
            "posneg_information_gain",
            "cumulative_posneg_information_gain_fraction",
        ]
        if column in posneg_table.columns
    ]
    return summary.merge(
        posneg_table[metadata_columns],
        on="sequence_order",
        how="left",
    )


def mark_summary_points(
    posneg_table,
    threshold_row=None,
    elbow=None,
    decision_rows=None,
):
    table = posneg_table.copy()
    table["is_90pct_cumulative_threshold"] = False
    table["is_elbow"] = False
    table["decision_change_count"] = 0
    if threshold_row is not None:
        table.loc[
            table["sequence_order"] == threshold_row["sequence_order"],
            "is_90pct_cumulative_threshold",
        ] = True
    if elbow is not None:
        table.loc[table["sequence_order"] == elbow["sequence_order"], "is_elbow"] = True
    if decision_rows is not None and not decision_rows.empty:
        counts = decision_rows["change_at_deleted_order"].value_counts()
        table["decision_change_count"] = (
            table["sequence_order"].map(counts).fillna(0).astype(int)
        )
    return table


def format_threshold_value(row, column):
    value = row.get(column)
    if pd.isna(value):
        return "NA"
    if column in {"sequence_order", "rank", "feature_index"}:
        return str(int(value))
    return str(value)


def plot_posneg_information_gain(
    posneg_table,
    output_path,
    threshold_row=None,
    elbow=None,
    decision_summary=None,
    threshold=0.9,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to write the plot.") from exc

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        posneg_table["sequence_order"],
        posneg_table["posneg_information_gain"],
        linewidth=1.4,
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Cancelled feature order")
    ax.set_ylabel("Posneg information gain")
    ax.set_title("Posneg Information Gain of Cancelled Features")
    ax.grid(True, alpha=0.3)

    if threshold_row is not None:
        threshold_order = threshold_row["sequence_order"]
        ax.axvline(
            threshold_order,
            color="tab:red",
            linewidth=1.2,
            linestyle="--",
            alpha=0.85,
        )
        ax.text(
            threshold_order,
            ax.get_ylim()[1],
            f"  {threshold:.0%} cumulative",
            color="tab:red",
            va="top",
        )

    if elbow is not None:
        elbow_order = elbow["sequence_order"]
        elbow_gain = elbow["posneg_information_gain"]
        elbow_fraction = elbow["cumulative_posneg_information_gain_fraction"] * 100
        ax.axvline(
            elbow_order,
            color="tab:green",
            linewidth=1.2,
            linestyle="--",
            alpha=0.9,
        )
        ax.scatter(
            [elbow_order],
            [elbow_gain],
            color="tab:green",
            s=36,
            zorder=5,
        )
        ax.text(
            elbow_order,
            elbow_gain,
            f"  elbow ({elbow_fraction:.1f}%)",
            color="tab:green",
            va="bottom",
        )

    if decision_summary is not None and not decision_summary.empty:
        plotted = decision_summary.dropna(subset=["sequence_order", "posneg_information_gain"])
        if not plotted.empty:
            ax.scatter(
                plotted["sequence_order"],
                plotted["posneg_information_gain"],
                color="tab:purple",
                marker="x",
                s=64,
                linewidths=1.8,
                zorder=6,
                label="Cervantes prediction change",
            )
            for _, row in plotted.iterrows():
                ax.axvline(
                    row["sequence_order"],
                    color="tab:purple",
                    linewidth=1.0,
                    linestyle="-.",
                    alpha=0.5,
                )
            ax.legend(loc="best")

    ax_fraction = ax.twinx()
    ax_fraction.plot(
        posneg_table["sequence_order"],
        posneg_table["cumulative_posneg_information_gain_fraction"] * 100,
        linewidth=1.2,
        linestyle=":",
        color="tab:orange",
    )
    ax_fraction.axhline(
        threshold * 100,
        color="tab:red",
        linewidth=1.0,
        linestyle="--",
        alpha=0.65,
    )
    if elbow is not None:
        elbow_fraction = elbow["cumulative_posneg_information_gain_fraction"] * 100
        ax_fraction.axhline(
            elbow_fraction,
            color="tab:green",
            linewidth=1.0,
            linestyle="--",
            alpha=0.65,
        )
    ax_fraction.set_ylabel("Cumulative posneg information gain (%)")
    ax_fraction.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    ablation_table = load_ablation_table(args.ablation_path)
    if "feature_index" not in ablation_table.columns:
        raise ValueError('Ablation table must contain a "feature_index" column.')

    if "posneg_information_gain" in ablation_table.columns:
        posneg_table = build_posneg_table(ablation_table)
    else:
        topic_matrix, y_quijote = prepare_topic_matrix(args)
        deleted_feature_indices = ablation_table["feature_index"].astype(int).tolist()
        posneg_information_gain_values = compute_deleted_posneg_information_gain(
            topic_matrix,
            y_quijote,
            deleted_feature_indices,
        )
        posneg_table = build_posneg_table(ablation_table, posneg_information_gain_values)

    decision_table = load_optional_table(args.decision_changes_path)
    decision_rows = decision_change_rows(
        decision_table,
        classifier_author=args.positive_author,
    )
    decision_summary = decision_change_summary(posneg_table, decision_rows)

    threshold_row = cumulative_threshold_row(posneg_table, threshold=0.9)
    elbow = elbow_row(posneg_table)
    posneg_table = mark_summary_points(
        posneg_table,
        threshold_row=threshold_row,
        elbow=elbow,
        decision_rows=decision_rows,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_prefix}.csv"
    json_path = output_dir / f"{args.output_prefix}.json"
    plot_path = output_dir / f"{args.output_prefix}.png"

    posneg_table.to_csv(csv_path, index=False)
    posneg_table.to_json(json_path, orient="records", indent=4)
    plot_posneg_information_gain(
        posneg_table,
        plot_path,
        threshold_row=threshold_row,
        elbow=elbow,
        decision_summary=decision_summary,
    )

    print(f"Saved posneg information gain table to {csv_path}")
    print(f"Saved posneg information gain JSON to {json_path}")
    print(f"Saved posneg information gain plot to {plot_path}")
    if threshold_row is not None:
        print(
            "90% cumulative posneg information gain reached at "
            f"sequence_order={format_threshold_value(threshold_row, 'sequence_order')}, "
            f"rank={format_threshold_value(threshold_row, 'rank')}, "
            f"feature_index={format_threshold_value(threshold_row, 'feature_index')}, "
            f"feature_name={format_threshold_value(threshold_row, 'feature_name')}"
        )
    else:
        print("90% cumulative posneg information gain threshold was not reached.")
    if elbow is not None:
        print(
            "Elbow found at "
            f"sequence_order={format_threshold_value(elbow, 'sequence_order')}, "
            f"rank={format_threshold_value(elbow, 'rank')}, "
            f"feature_index={format_threshold_value(elbow, 'feature_index')}, "
            f"feature_name={format_threshold_value(elbow, 'feature_name')}, "
            "cumulative_posneg_information_gain="
            f"{elbow['cumulative_posneg_information_gain_fraction'] * 100:.2f}%"
        )
    else:
        print("Elbow could not be identified.")
    if not decision_summary.empty:
        print(f"{args.positive_author} prediction changes:")
        for _, row in decision_summary.iterrows():
            print(
                "  "
                f"sequence_order={format_threshold_value(row, 'sequence_order')}, "
                f"rank={format_threshold_value(row, 'rank')}, "
                f"feature_index={format_threshold_value(row, 'feature_index')}, "
                f"feature_name={format_threshold_value(row, 'feature_name')}, "
                f"decision_change_count={int(row['decision_change_count'])}"
            )
    else:
        print(f"No {args.positive_author} decision changes found to plot.")


if __name__ == "__main__":
    main()
