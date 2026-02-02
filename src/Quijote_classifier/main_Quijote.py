import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spacy
import pandas as pd
import json

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from Quijote_classifier.quijote_classifier import TextClassificationTrainer
from src.data_preparation.data_loader import load_corpus, binarize_title
# from Quijote_classifier import TextClassificationTrainer
from src.feature_extraction.features import FeaturesFrequentWords
from supervised_term_weighting.tsr_functions import (
    get_supervised_matrix, get_tsr_matrix, posneg_information_gain, gss
)

warnings.filterwarnings("ignore")


@dataclass
class ModelConfig:
    """Configuration for the model training and evaluation"""
    n_jobs: int = 30
    segment_min_token_size: int = 500
    random_state: int = 0
    max_features: int = 3000
    save_res: bool = True
    results_inference: str = 'inference_results.csv'
    classifier_type: str = "lr"
    # Logistic Regression hyperparameters
    C: float = 1.0
    penalty: str = 'l2'
    solver: str = 'lbfgs'
    class_weight: str = 'balanced'
    # Add these attributes
    train_dir: str = '../../corpus/quixote_vs_notquixote'
    # test_dir: str = '../../corpus/test'
    target_title: str = 'Quijote'
    feature_importance_file: str = 'feature_importance.json'

    @classmethod
    def from_args(cls):
        """Create config from command line args"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--train-dir', default='../../corpus/quixote_vs_notquixote')
        parser.add_argument('--target-title', default='Quijote')
        parser.add_argument('--max-features', type=int, default=10_000,
                            help='Number of most frequent words to use')
        parser.add_argument('--results-inference', default='../../results/Quijote_class/results_inference.csv',
                            help='Filename for saving results')
        parser.add_argument('--feature-importance', default='../../results/Quijote_class/feature_importance.json',
                            help='Filename for saving feature importance')

        args = parser.parse_args()

        # Create config instance first
        config = cls()
        config.train_dir = args.train_dir
        # config.test_dir = args.test_dir
        config.max_features = args.max_features
        config.results_inference = str(Path(args.results_inference).parent /
                                       f"results_Quijote.csv")
        config.feature_importance_file = str(Path(args.feature_importance).parent /
                                             f"feature_importance_Quijote.json")
        config.target_title = args.target_title

        Path(config.results_inference).parent.mkdir(parents=True, exist_ok=True)
        return config


if __name__ == '__main__':
    config = ModelConfig.from_args()

    train_corpus = load_corpus(config.train_dir, cache_path='../data_preparation/.cache')
    train_corpus = binarize_title(train_corpus, config.target_title)
    # test_corpus = load_corpus(config.test_dir, cache_path='../data_preparation/.cache')
    # test_corpus = binarize_title(test_corpus, config.target_title)

    trainer = TextClassificationTrainer(
        max_features=config.max_features,
        target_title=config.target_title,
        C=config.C,
        penalty=config.penalty,
        solver=config.solver,
        class_weight=config.class_weight,
        random_state=config.random_state,
        n_jobs=config.n_jobs
    )

    documents, y = trainer._prepare_training_data(train_corpus)

    X = trainer.vectorizer.fit_transform(documents)
    print(f'done {X.shape}')

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    tsr_metric = posneg_information_gain
    # tsr_metric = gss
    label_matrix = np.asarray(ytr).reshape(-1, 1)
    supervised_matrix = get_supervised_matrix(Xtr, label_matrix, n_jobs=-1)
    tsr_matrix = get_tsr_matrix(supervised_matrix, tsr_metric, n_jobs=-1).flatten()
    feat_idx_importance = np.argsort(tsr_matrix)[::-1]
    feat_idx_importance = [idx for idx in feat_idx_importance if tsr_matrix[idx]>0]
    vocabulary = trainer.vectorizer.vectorizer.get_feature_names_out()

    # lr = LogisticRegression()
    lr = GridSearchCV(LogisticRegression(),
                      param_grid={
                          'C':np.logspace(-3,3,7),
                          'class_weight':['balanced', None],
                      },
                      n_jobs=-1)


    feats_used = Xtr.shape[1]
    remove_at_loop = 10
    delete_pointer = 0

    print(f'prevalence Quixote"s": {np.mean(ytr)*100:.3f}%')

    # loop until the classifier degenerates
    degenerated = False
    candidates = True
    # first = True
    while not degenerated and candidates:
        # if first:
        lr.fit(Xtr, ytr)
            # first = False
        yte_hat = lr.predict(Xte)
        acc = (yte==yte_hat).mean()
        f1 = f1_score(yte, yte_hat, )
        print(f'F1={100 * f1:.2f}% Acc={acc*100:.2f}% num-feats={feats_used}')

        if f1 <= 0.55:
            degenerated = True
            print('stop: classifier has degenerated')
        elif delete_pointer < len(feat_idx_importance):
            to_delete = feat_idx_importance[delete_pointer:delete_pointer+remove_at_loop]
            Xtr[:, to_delete] = 0
            Xte[:, to_delete] = 0
            Xtr.eliminate_zeros()
            Xte.eliminate_zeros()
            Xtr = normalize(Xtr, norm='l2', axis=1)
            Xte = normalize(Xte, norm='l2', axis=1)
            delete_pointer+=remove_at_loop
            feats_used-=remove_at_loop
            print(f'deleting: {[f"{vocabulary[i]} ({tsr_matrix[i]:.2f})" for i in to_delete]}')
        else:
            candidates = False
            print('stop: no more candidates to remove')


