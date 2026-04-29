import argparse
from pathlib import Path
from types import SimpleNamespace
import pickle

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION_PATH = PROJECT_ROOT / "results" / "inference" / "ablation.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "inference"
DEFAULT_TRAIN_DIR = (
    PROJECT_ROOT / "corpus" / "training"
    if (PROJECT_ROOT / "corpus" / "training").exists()
    else PROJECT_ROOT / "corpus"
)
DEFAULT_HYPERPARAMS_PATH = PROJECT_ROOT / "hyperparams" / "hyperparameters.pkl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot cumulative information gain for deleted ablation features."
    )
    parser.add_argument("--ablation-path", default=str(DEFAULT_ABLATION_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-prefix", default="ablation_cumulative_information_gain")
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


def compute_deleted_information_gain(topic_matrix, y_quijote, deleted_feature_indices):
    from quijote_classifier.supervised_term_weighting.tsr_functions import (
        get_supervised_matrix,
        get_tsr_matrix,
        information_gain,
    )

    deleted_matrix = topic_matrix[:, deleted_feature_indices]
    label_matrix = np.asarray(y_quijote).reshape(-1, 1)
    supervised_matrix = get_supervised_matrix(deleted_matrix, label_matrix, n_jobs=-1)
    return get_tsr_matrix(supervised_matrix, information_gain, n_jobs=-1).flatten()


def build_cumulative_table(ablation_table, information_gain_values):
    table = ablation_table.copy()
    table["information_gain"] = information_gain_values
    table = table.sort_values(
        ["information_gain", "rank", "feature_index"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    table["ascending_information_gain_order"] = np.arange(1, len(table) + 1)
    table["cumulative_information_gain"] = table["information_gain"].cumsum()
    total_information_gain = table["information_gain"].sum()
    table["cumulative_information_gain_fraction"] = (
        table["cumulative_information_gain"] / total_information_gain
        if total_information_gain
        else 0.0
    )
    return table


def plot_cumulative_information_gain(cumulative_table, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to write the plot.") from exc

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        cumulative_table["ascending_information_gain_order"],
        cumulative_table["cumulative_information_gain"],
        linewidth=2,
    )
    ax.set_xlabel("Cancelled features, sorted from lowest to highest information gain")
    ax.set_ylabel("Cumulative information gain")
    ax.set_title("Cumulative Information Gain of Cancelled Features")
    ax.grid(True, alpha=0.3)

    ax_fraction = ax.twinx()
    ax_fraction.plot(
        cumulative_table["ascending_information_gain_order"],
        cumulative_table["cumulative_information_gain_fraction"] * 100,
        linewidth=1.5,
        linestyle="--",
        color="tab:orange",
    )
    ax_fraction.set_ylabel("Cumulative information gain (%)")
    ax_fraction.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    ablation_table = load_ablation_table(args.ablation_path)
    if "feature_index" not in ablation_table.columns:
        raise ValueError('Ablation table must contain a "feature_index" column.')

    topic_matrix, y_quijote = prepare_topic_matrix(args)
    deleted_feature_indices = ablation_table["feature_index"].astype(int).tolist()
    information_gain_values = compute_deleted_information_gain(
        topic_matrix,
        y_quijote,
        deleted_feature_indices,
    )
    cumulative_table = build_cumulative_table(ablation_table, information_gain_values)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_prefix}.csv"
    json_path = output_dir / f"{args.output_prefix}.json"
    plot_path = output_dir / f"{args.output_prefix}.png"

    cumulative_table.to_csv(csv_path, index=False)
    cumulative_table.to_json(json_path, orient="records", indent=4)
    plot_cumulative_information_gain(cumulative_table, plot_path)

    print(f"Saved cumulative information gain table to {csv_path}")
    print(f"Saved cumulative information gain JSON to {json_path}")
    print(f"Saved cumulative information gain plot to {plot_path}")


if __name__ == "__main__":
    main()
