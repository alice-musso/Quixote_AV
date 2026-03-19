import os.path
import pickle
import sys
import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import spacy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, LeaveOneGroupOut

from commons import AuthorshipVerification
from data_preparation.data_loader import load_corpus, binarize_corpus, binarize_title
from Quijote_classifier.QuijotevsnotQuijiote_experiment import binarize_labels_for_topic, ablation, compute_feature_ranking
from Quijote_classifier.supervised_term_weighting.tsr_functions import posneg_information_gain, gss, chi_square

warnings.filterwarnings("ignore")


@dataclass
class ModelConfig:
    """Configuration for the model training and evaluation"""
    n_jobs: int = -1
    segment_min_token_size: int = 500
    random_state: int = 0
    max_features: int = 5000
    oversample: bool = False
    rebalance_ratio: float = 0.5
    save_res: bool = True
    results_inference: str = 'inference_results.csv'
    results_loo: str = 'loo_results.csv'
    hyperparams_save: str = 'hyperparameters_posauth_Cervantes.pkl'
    classifier_type: str = "lr"

    @classmethod
    def from_args(cls):
        """Create config from command line args"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--train-dir', default='../corpus/training')
        parser.add_argument('--test-dir', default='../corpus/test')
        parser.add_argument('--positive-author', default='Cervantes',
                            help='If indicated (default: Cervantes), binarizes the corpus, '
                                 'otherwise assumes multiclass classification')
        parser.add_argument('--results-inference', default='../results/inference/results.csv',
                            help='Filename for saving results')
        parser.add_argument('--results-loo', default='../results/loo/results.csv',
                            help='Filename for saving results for the leave one out whole books + segments')
        parser.add_argument('--hyperparams-save', default='../hyperparams/hyperparameters_posauth_Cervantes.pkl')
        parser.add_argument('--classifier-type', choices=["lr", "svm"], default='lr')
        parser.add_argument('--load-hyperparams', default=True, action='store_false')

        args = parser.parse_args()

        if '--target' in sys.argv and '--test-document' not in sys.argv:
            args.test_document = ''

        config = cls()
        config.train_dir = args.train_dir
        config.test_dir = args.test_dir
        config.positive_author = args.positive_author
        config.classifier_type = args.classifier_type
        config.results_inference = str(Path(args.results_inference).parent /
                                       f"results_{config.positive_author}_{config.classifier_type}.csv")
        config.results_loo = str(Path(args.results_loo).parent /
                                 f"loo_results_{config.positive_author}_{config.classifier_type}.csv")
        config.hyperparams_save = str(Path(args.hyperparams_save).parent / f"hyperparameters_posauth_Cervantes.pkl")
        config.load_hyperparams = args.load_hyperparams
        for path in [config.results_inference, config.results_loo, config.hyperparams_save]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        return config


if __name__ == '__main__':
    config = ModelConfig.from_args()

    train_corpus = load_corpus(config.train_dir)
    test_corpus = load_corpus(config.test_dir)

    if config.positive_author:
        train_corpus = binarize_corpus(train_corpus, positive_author=config.positive_author)
        test_corpus = binarize_corpus(test_corpus, positive_author=config.positive_author)

    spacy_language_model = spacy.load('es_dep_news_trf')
    av_system = AuthorshipVerification(config, nlp=spacy_language_model)

    # Feature-block selection for authorship verification ('Cervantes')
    # --------------------------------------------------------------------------------------------
    if not config.load_hyperparams:
        X_select, X_test_select, y, groups, best_params, best_score = av_system.model_selection(
            train_corpus, test_corpus, save_hyper_path=config.hyperparams_save, refit=False
        )
    else:
        hyper_path = Path(config.hyperparams_save)
        if not hyper_path.exists():
            raise FileNotFoundError(f"{hyper_path} does not exist")
        with hyper_path.open("rb") as f:
            hyperparams = pickle.load(f)
        X_select, y, slices, groups, best_params, best_score = av_system.fit_with_hyperparams(train_corpus,
                                                                                       hyperparams=hyperparams)
    #else:
        #raise NotImplementedError('not yet revised')

    # Feature ablation for topic removal ('Quixote')
    # --------------------------------------------------------------------------------------------
    documents, y_quixote, groups = binarize_labels_for_topic(train_corpus, target_title="Quijote")
    feat_idx_importance, tsr_matrix = compute_feature_ranking(X_select, y_quixote, tsr_metric=posneg_information_gain)

    best_cls_params = {'C': best_params['C'], 'class_weight': best_params['class_weight']}
    classifier = av_system.new_classifier().set_params(**best_cls_params)
    X_clean, X_test_clean = ablation(feat_idx_importance, tsr_matrix, X_select, X_test_select, y_quixote, groups, classifier)

    # Leave-one-out performance check for authorship verification ('Cervantes') after ablation
    # --------------------------------------------------------------------------------------------
    f1_score_post_cleaning = cross_val_score(
        estimator=av_system.new_classifier(),
        X=X_clean,
        y=y,
        groups=groups,
        cv=LeaveOneGroupOut(),
        scoring=make_scorer(f1_score, pos_label=av_system.config.positive_author, zero_division=1.0),
        n_jobs=-1
    )
    f1_score_post_cleaning = np.mean(f1_score_post_cleaning)

    print(f'F1-pre-clean:  {best_score:.4f}')
    print(f'F1-post-clean: {f1_score_post_cleaning:.4f}')


    # final prediction
    # --------------------------------------------------------------------------------------------
    classifier = av_system.new_classifier().set_params(**best_cls_params)
    calibrated_classifier = CalibratedClassifierCV(
        classifier,
        cv=10,
        # method="isotonic",
        method="sigmoid",
        n_jobs=-1,
    ).fit(X_clean, y)

    y_probs = calibrated_classifier.predict_proba(X_test_clean)
    for y_probs_i, book in zip(y_probs, test_corpus):
        print(f'title={book.title}: got posterior = {y_probs_i}')
