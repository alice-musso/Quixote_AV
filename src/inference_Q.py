"""
Pipeline: De-Quijotized Authorship Verification
================================================
Steps:
  1. Load the unique train corpus (same dir for both tasks).
     Extract the full feature matrix X and find the best feature slices
     via GridSearchCV (as in main_inference.py).
  2. Build Quijote/notQuijote binary labels from the same corpus and
     compute chi-square TSR scores over the FULL feature matrix X
     (all feature types, not just frequent words).
     Iteratively zero out the most discriminative columns until the
     Quijote classifier degenerates (accuracy ≤ threshold).
  3. Re-fit the inference classifier on X_clean (de-quijotized matrix)
     using the best hyperparameters from step 1.
     Run LOO evaluation and inference on the test set.
"""

import argparse
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import spacy
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import normalize

from commons import AuthorshipVerification
from data_preparation.data_loader import load_corpus, binarize_corpus
from src.Quijote_classifier.supervised_term_weighting.tsr_functions import (
    chi_square,
    get_supervised_matrix,
    get_tsr_matrix,
)
from src.Quijote_classifier.QuijotevsnotQuijiote_experiment import ablation

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Unified configuration for the de-quijotized authorship pipeline."""

    # Paths  (train_dir is shared between the Quijote task and the author task)
    train_dir: str = "../corpus/training"
    test_dir: str = "../corpus/test"
    results_inference: str = "../results/inference/results_dequijotized.csv"
    results_loo: str = "../results/loo/loo_results_dequijotized.csv"
    hyperparams_save: str = "../hyperparams/hyperparameters_dequijotized.pkl"

    # Author-attribution task
    positive_author: str = "Cervantes"
    classifier_type: str = "lr"

    # Quijote ablation
    target_title: str = "Quijote"
    quijote_remove_per_step: int = 10
    quijote_degeneration_threshold: float = 0.55

    # Misc
    n_jobs: int = -1
    random_state: int = 0
    max_features: int = 5000

    @classmethod
    def from_args(cls) -> "ModelConfig":
        parser = argparse.ArgumentParser(
            description="De-quijotized authorship verification pipeline"
        )
        parser.add_argument("--train-dir", default="../corpus/training",
                            help="Training corpus (shared for author and Quijote tasks)")
        parser.add_argument("--test-dir", default="../corpus/test")
        parser.add_argument("--positive-author", default="Cervantes")
        parser.add_argument("--target-title", default="Quijote",
                            help="Title used as positive class in the ablation step")
        parser.add_argument("--results-inference",
                            default="../results/inference/results_dequijotized.csv")
        parser.add_argument("--results-loo",
                            default="../results/loo/loo_results_dequijotized.csv")
        parser.add_argument("--hyperparams-save",
                            default="../hyperparams/hyperparameters_dequijotized.pkl")
        parser.add_argument("--classifier-type", choices=["lr", "svm"], default="lr")
        parser.add_argument("--max-features", type=int, default=5000)
        parser.add_argument("--remove-per-step", type=int, default=10,
                            help="Columns removed per ablation iteration")
        parser.add_argument("--degeneration-threshold", type=float, default=0.55,
                            help="Accuracy below which the Quijote classifier is degenerated")

        args = parser.parse_args()
        cfg = cls()
        cfg.train_dir          = args.train_dir
        cfg.test_dir           = args.test_dir
        cfg.positive_author    = args.positive_author
        cfg.target_title       = args.target_title
        cfg.classifier_type    = args.classifier_type
        cfg.max_features       = args.max_features
        cfg.quijote_remove_per_step         = args.remove_per_step
        cfg.quijote_degeneration_threshold  = args.degeneration_threshold

        suffix = f"{cfg.positive_author}_{cfg.classifier_type}_dequijotized"
        cfg.results_inference = str(
            Path(args.results_inference).parent / f"results_{suffix}.csv"
        )
        cfg.results_loo = str(
            Path(args.results_loo).parent / f"loo_{suffix}.csv"
        )
        cfg.hyperparams_save = str(
            Path(args.hyperparams_save).parent / f"hyperparams_{suffix}.pkl"
        )
        for path in [cfg.results_inference, cfg.results_loo, cfg.hyperparams_save]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        return cfg

# ---------------------------------------------------------------------------
# Step 2 — Build Quijote labels (same corpus, same row order as X)
# ---------------------------------------------------------------------------

def build_quijote_labels(train_corpus, target_title: str):
    """
    Build Quijote/notQuijote binary labels aligned row-by-row with X.

    av_system.prepare_X_y() iterates:
        book.processed → book.segment[0] → book.segment[1] …
    for every book, so we replicate exactly the same order here.

    Returns
    -------
    y_quijote      : np.ndarray of str, shape (n_samples,)
    groups_quijote : np.ndarray of int  (identical to 'groups' from step 1)
    unique_titles  : list[str]  one title per group, for reporting
    """
    negative_label = f"Not{target_title}"
    y_quijote, groups, titles_per_row = [], [], []

    for i, book in enumerate(train_corpus):
        label = target_title if book.title == target_title else negative_label
        # full-book document
        y_quijote.append(label)
        groups.append(i)
        titles_per_row.append(book.title)
        # segments
        for _ in book.segmented:
            y_quijote.append(label)
            groups.append(i)
            titles_per_row.append(book.title)

    seen, unique_titles = set(), []
    for t in titles_per_row:
        if t not in seen:
            seen.add(t)
            unique_titles.append(t)

    return np.asarray(y_quijote), np.asarray(groups), unique_titles


# ---------------------------------------------------------------------------
# Step 2 — Chi-square ranking across ALL feature columns
# ---------------------------------------------------------------------------

def compute_full_feature_ranking(X, y_quijote):
    """
    Compute chi-square TSR scores over every column of X using
    Quijote/notQuijote binary labels.

    A 70/30 split is used for the supervised cell-count estimation,
    consistent with main_Quijote.py.

    Returns
    -------
    feat_idx_importance : list[int]  column indices sorted by score desc,
                          limited to those with score > 0
    tsr_scores          : np.ndarray  chi-square score per column (flat)
    """
    X_dense = X.toarray() if sp.issparse(X) else np.asarray(X)

    Xtr, _, ytr, _ = train_test_split(
        X_dense, y_quijote, test_size=0.3, random_state=0
    )
    label_matrix = np.asarray(ytr).reshape(-1, 1)
    supervised_matrix = get_supervised_matrix(Xtr, label_matrix, n_jobs=-1)
    tsr_scores = get_tsr_matrix(supervised_matrix, chi_square, n_jobs=-1).flatten()

    feat_idx_importance = list(np.argsort(tsr_scores)[::-1])
    feat_idx_importance = [i for i in feat_idx_importance if tsr_scores[i] > 0]
    return feat_idx_importance, tsr_scores


# ---------------------------------------------------------------------------
# Step 3 — Re-fit inference classifier on de-quijotized X
# ---------------------------------------------------------------------------

def fit_dequijotized_classifier(
    av_system: AuthorshipVerification,
    X_clean,
    y: np.ndarray,
    groups,
    best_params: dict,
    train_corpus,
    test_corpus,
    config: ModelConfig,
):
    """
    Fit ClassifierRange with the winning hyperparameters (from step 1) on
    X_clean, run LOO evaluation, then predict on the test corpus.
    """
    print("\n" + "=" * 60)
    print("STEP 3 — Re-fitting inference classifier on de-quijotized matrix")
    print("=" * 60)

    cls_range = av_system.prepare_classifier()
    av_system.best_params = best_params
    av_system.best_score  = None
    av_system.fit_classifier_range(X_clean, y, cls_range, best_params)

    print("\nRunning leave-one-out evaluation …")
    av_system.leave_one_out(train_corpus)

    print("\nRunning inference on test corpus …")
    predicted_authors, posteriors = av_system.predict(
        test_corpus, return_posteriors=True
    )
    pos_idx = av_system.index_of_author(config.positive_author)

    print("\nInference results:")
    for i, book in enumerate(test_corpus):
        print(
            f'  "{book.title}"  author={book.author}  '
            f"predicted={predicted_authors[i]}  "
            f"posterior={posteriors[i, pos_idx]:.4f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = ModelConfig.from_args()

    # ------------------------------------------------------------------ #
    #  Load corpora  (single shared train_dir)                            #
    # ------------------------------------------------------------------ #
    print("Loading corpora …")
    train_corpus = load_corpus(config.train_dir)
    test_corpus  = load_corpus(config.test_dir)

    train_corpus = binarize_corpus(train_corpus, positive_author=config.positive_author)
    test_corpus  = binarize_corpus(test_corpus,  positive_author=config.positive_author)

    spacy_nlp = spacy.load("es_dep_news_trf")
    av_system = AuthorshipVerification(config, nlp=spacy_nlp)


    X, y, slices, groups, best_params, best_score = av_system.fit(train_corpus=train_corpus,
        save_hyper_path=config.hyperparams_save)



    #y_quijote, groups_quijote, unique_titles = build_quijote_labels(
    #    train_corpus=train_corpus,
    #    target_title=config.target_title,
    #)

    X_clean = ablation( train_corpus=train_corpus)


    fit_dequijotized_classifier(
        av_system=av_system,
        X_clean=X_clean,
        y=y,
        groups=groups,
        best_params=best_params,
        train_corpus=train_corpus,
        test_corpus=test_corpus,
        config=config,
    )
