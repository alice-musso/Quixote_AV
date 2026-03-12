import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneGroupOut, LeaveOneOut, cross_val_score
from sklearn.preprocessing import normalize
from array import array
from src.Quijote_classifier.supervised_term_weighting.tsr_functions import chi_square, get_supervised_matrix, get_tsr_matrix, posneg_information_gain, gss
from src.data_preparation.data_loader import load_corpus, binarize_title
# from Quijote_classifier import TextClassificationTrainer
from src.feature_extraction.features import FeaturesFrequentWords
from src.data_preparation.data_loader import Book, get_spanish_function_words
from typing import List, Dict, Optional

warnings.filterwarnings("ignore")


def binarize_labels_for_topic(books: List[Book], target_title):
    """
    Extract texts and binary labels from Book objects.
    """
    documents = []
    labels = []
    groups = []

    for g, book in enumerate(books):
        if target_title.lower() in book.title.lower():
            label = 1
        else:
            label = 0

        # Full book
        documents.append(book.processed)
        labels.append(label)
        groups.append(g)

        # Segments (same label)
        if book.segmented is not None:
            for fragment in book.segmented:
                documents.append(fragment)
                labels.append(label)
                groups.append(g)

    return documents, labels, groups

def compute_feature_ranking(X, y, tsr_metric):

    Xtr, Xte, ytr, yte = (
        train_test_split(X, y, test_size=0.3, random_state=0)
    )

    label_matrix = np.asarray(ytr).reshape(-1, 1)
    supervised_matrix = get_supervised_matrix(Xtr, label_matrix, n_jobs=-1)
    tsr_matrix = get_tsr_matrix(supervised_matrix, tsr_metric, n_jobs=-1).flatten()
    feat_idx_importance = np.argsort(tsr_matrix)[::-1]
    feat_idx_importance = [idx for idx in feat_idx_importance if tsr_matrix[idx] > 0]
    return feat_idx_importance, tsr_matrix

def ablation(feat_idx_importance, vocabulary, tsr_matrix, X, y, groups):

    feats_used = X.shape[1]
    remove_at_loop = 10
    delete_pointer = 0

    print(f'prevalence Quixote"s": {np.mean(y) * 100:.3f}%')

    # loop until the classifier degenerates
    degenerated = False
    candidates = True
    # first = True
    X = X.toarray()
    titles = [b.title for b in train_corpus]
    # label_assignments = [(title, label) for title, label in zip(titles, y)]
    # print(f'{label_assignments=}')

    # loo = LeaveOneGroupOut()
    # loo.get_n_splits()
    while not degenerated and candidates:
        acc = cross_val_score(
            estimator=LogisticRegressionCV(),
            X=X, y=y,
            cv=LeaveOneGroupOut(),
            groups=groups,
            scoring='accuracy',
            n_jobs=-1,
        )

        for (acc_i, title_i) in zip(acc, titles):
            print(f'classification accuracy for {title_i} is {acc_i * 100:.2f}%')
        acc = acc.mean()

        print(f'Acc={acc * 100:.2f}% num-feats={feats_used}')

        if acc <= 0.55:
            degenerated = True
            print('stop: classifier has degenerated')
        elif delete_pointer < len(feat_idx_importance):
            to_delete = feat_idx_importance[delete_pointer:delete_pointer + remove_at_loop]
            X[:, to_delete] = 0

            # X.eliminate_zeros()
            X = normalize(X, norm='l2', axis=1)
            # Xte = normalize(Xte, norm='l2', axis=1)
            delete_pointer += remove_at_loop
            feats_used -= remove_at_loop
            print(f'deleting: {[f"{vocabulary[i]} ({tsr_matrix[i]:.2f})" for i in to_delete]}')
        else:
            candidates = False
            print('stop: no more candidates to remove')

            return X

@dataclass
class Config:
    train_dir: str = '../../corpus/training'
    target_title: str = 'Quijote'

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train-dir', default=cls.train_dir)
        parser.add_argument('--target-title', default=cls.target_title)
        args = parser.parse_args()
        return cls(**vars(args))

if __name__ == '__main__':
    config = Config.from_args()

    train_corpus = load_corpus(config.train_dir, cache_path='../data_preparation/.cache')
    train_corpus = binarize_title(train_corpus, target_title=config.target_title)

    documents, y, groups = binarize_labels_for_topic(train_corpus)
    vectorizer = FeaturesFrequentWords(max_features=3000,
        remove_stopwords=list(get_spanish_function_words())
    )
    X = vectorizer.fit_transform(documents)
    print(f'done {X.shape}')
    vocabulary = vectorizer.vectorizer.get_feature_names_out()

    tsr_metric = posneg_information_gain
    #tsr_metric = gss
    #tsr_metric = chi_square

    feat_idx_importance, tsr_matrix = compute_feature_ranking(X, y, tsr_metric)

    ablation(feat_idx_importance, vocabulary, tsr_matrix, X, y, groups)