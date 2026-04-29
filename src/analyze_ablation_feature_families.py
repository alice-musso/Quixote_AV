import argparse
from pathlib import Path
from types import SimpleNamespace
import pickle

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION_PATH = PROJECT_ROOT / "results" / "inference" / "ablation.csv"
DEFAULT_TRAIN_DIR = (
    PROJECT_ROOT / "corpus" / "training"
    if (PROJECT_ROOT / "corpus" / "training").exists()
    else PROJECT_ROOT / "corpus"
)
DEFAULT_HYPERPARAMS_PATH = PROJECT_ROOT / "hyperparams" / "hyperparameters.pkl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize deleted ablation features by feature family."
    )
    parser.add_argument(
        "--ablation-path",
        default=str(DEFAULT_ABLATION_PATH),
        help="Path to ablation.csv or ablation.json.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Supports .csv and .json.",
    )
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--positive-author", default="Cervantes")
    parser.add_argument("--classifier-type", choices=["lr", "svm"], default="lr")
    parser.add_argument("--hyperparams-save", default=str(DEFAULT_HYPERPARAMS_PATH))
    parser.add_argument(
        "--load-hyperparams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load saved hyperparameters. Use --no-load-hyperparams to rerun "
            "model selection and save the selected hyperparameters."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument(
        "--family-size",
        action="append",
        default=[],
        metavar="FAMILY=SIZE",
        help=(
            "Optional exact denominator for a feature family, e.g. "
            "--family-size feat_char=5000. Can be repeated."
        ),
    )
    return parser.parse_args()


def load_ablation_table(path):
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def feature_family(feature_name):
    feature_name = str(feature_name)
    if ":" not in feature_name:
        return "unknown"
    return feature_name.split(":", 1)[0]


def parse_family_sizes(family_size_args):
    family_sizes = {}
    for item in family_size_args:
        if "=" not in item:
            raise ValueError(f'Family size must use FAMILY=SIZE format: "{item}"')
        family, size = item.split("=", 1)
        family = family.strip()
        if not family:
            raise ValueError(f'Family name cannot be empty: "{item}"')
        family_sizes[family] = int(size)
    return family_sizes


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


def recompute_family_sizes(args):
    from authorship_verification import AuthorshipVerification
    from data_preparation.data_loader import binarize_corpus, load_corpus

    hyperparams_path = resolve_hyperparams_path(args)
    hyperparams = None
    if args.load_hyperparams and not hyperparams_path.exists():
        raise RuntimeError(
            f"{hyperparams_path} does not exist. Run inference once to save "
            "hyperparameters, pass --no-load-hyperparams to rerun model "
            "selection, or pass --family-size FAMILY=SIZE overrides."
        )
    if args.load_hyperparams:
        hyperparams = load_hyperparams(hyperparams_path)

    verifier = AuthorshipVerification(build_verifier_config(args))
    train_corpus = binarize_corpus(
        load_corpus(
            args.train_dir,
            cache_path=str(PROJECT_ROOT / "src" / "data_preparation" / ".cache"),
        ),
        positive_author=args.positive_author,
    )
    verifier_artifacts = verifier.prepare_verifier(
        train_documents=train_corpus,
        test_documents=train_corpus,
        hyperparams=hyperparams,
        save_hyper_path=None if args.load_hyperparams else hyperparams_path,
    )
    selected_feature_names = pd.Series(
        verifier_artifacts.feature_selection.selected_feature_names,
        name="feature_name",
    )
    return selected_feature_names.map(feature_family).value_counts().to_dict()


def build_family_summary(ablation_table, family_sizes):
    if "feature_name" not in ablation_table.columns:
        raise ValueError('Ablation table must contain a "feature_name" column.')
    if "feature_index" not in ablation_table.columns:
        raise ValueError('Ablation table must contain a "feature_index" column.')

    ablation_table = ablation_table.copy()
    ablation_table["feature_family"] = ablation_table["feature_name"].map(feature_family)
    family_counts = (
        ablation_table["feature_family"]
        .value_counts()
        .rename_axis("feature_family")
        .reset_index(name="cancelled_features_in_family")
    )
    family_size_table = pd.DataFrame(
        [
            {"feature_family": family, "family_total_features": size}
            for family, size in family_sizes.items()
        ]
    )
    family_counts = family_counts.merge(family_size_table, on="feature_family", how="left")
    if family_counts["family_total_features"].isna().any():
        missing = family_counts.loc[
            family_counts["family_total_features"].isna(),
            "feature_family",
        ].tolist()
        raise ValueError(
            "Missing total feature-family sizes for: "
            + ", ".join(missing)
        )

    family_counts["cancelled_family_fraction"] = (
        family_counts["cancelled_features_in_family"]
        / family_counts["family_total_features"]
    )
    family_counts["cancelled_family_percentage"] = (
        family_counts["cancelled_family_fraction"] * 100
    )
    family_counts = (
        family_counts.sort_values(
            ["cancelled_family_fraction", "feature_family"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    return family_counts


def save_summary(summary, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".json":
        summary.to_json(output_path, orient="records", indent=4)
    else:
        summary.to_csv(output_path, index=False)


def main():
    args = parse_args()
    ablation_table = load_ablation_table(args.ablation_path)
    family_sizes = recompute_family_sizes(args)
    family_sizes.update(parse_family_sizes(args.family_size))
    summary = build_family_summary(
        ablation_table,
        family_sizes=family_sizes,
    )
    print(summary.to_string(index=False))
    if args.output:
        save_summary(summary, args.output)
        print(f"\nSaved feature-family summary to {args.output}")


if __name__ == "__main__":
    main()
