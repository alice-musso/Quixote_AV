import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    train_dir: str
    test_dir: str
    positive_author: str
    target_title: str
    results_inference: str
    hyperparams_save: str
    classifier_type: str
    load_hyperparams: bool
    skip_ablation: bool = False
    skip_decision_changes: bool = False
    n_jobs: int = -1
    random_state: int = 0
    max_features: int = 5000

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--train-dir", default="../corpus/training")
        parser.add_argument("--test-dir", default="../corpus/test")
        parser.add_argument("--positive-author", default="Cervantes")
        parser.add_argument("--target-title", default="Quijote")
        parser.add_argument(
            "--results-inference",
            default="../results/inference/results.csv",
            help="Directory anchor for generated JSON and score files.",
        )
        parser.add_argument(
            "--hyperparams-save",
            default="../hyperparams/hyperparameters.pkl",
        )
        parser.add_argument("--classifier-type", choices=["lr", "svm"], default="lr")
        parser.add_argument(
            "--load-hyperparams",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Load saved hyperparameters; use --no-load-hyperparams to rerun model selection.",
        )
        parser.add_argument(
            "--skip-ablation",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Skip the topic-ablation step; use --no-skip-ablation to re-enable it.",
        )
        parser.add_argument(
            "--skip-decision-changes",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Skip decision-change tracing for a faster run.",
        )
        args = parser.parse_args()
        return cls.from_namespace(args)

    @classmethod
    def from_namespace(cls, args):
        positive_author = args.positive_author
        classifier_type = args.classifier_type
        results_inference = str(
            Path(args.results_inference).parent
            / f"results_{positive_author}_{classifier_type}.json"
        )
        hyperparams_save = str(
            Path(args.hyperparams_save).parent
            / f"hyperparameters_posauth_{positive_author}.pkl"
        )

        config = cls(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            positive_author=positive_author,
            target_title=args.target_title,
            results_inference=results_inference,
            hyperparams_save=hyperparams_save,
            classifier_type=classifier_type,
            load_hyperparams=args.load_hyperparams,
            skip_ablation=args.skip_ablation,
            skip_decision_changes=args.skip_decision_changes,
        )
        config.ensure_output_dirs()
        return config

    def ensure_output_dirs(self):
        for path in [self.results_inference, self.hyperparams_save]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
