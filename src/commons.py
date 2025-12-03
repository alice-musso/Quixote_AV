import pickle
from typing import List, Tuple, Dict, Optional, Any, Union
import numpy as np
import spacy
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import (LogisticRegression)
from sklearn.model_selection import (StratifiedGroupKFold,
                                     GridSearchCV,
                                     LeaveOneGroupOut,
                                     )
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    f1_score, 
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    make_scorer
)
import csv
import time
from pathlib import Path

from torch.backends.cudnn import flags

from data_preparation.data_loader import (binarize_corpus,
                                            load_corpus,
                                            get_spanish_function_words,
                                            Book)

from feature_extraction.features import (
        FeaturesFunctionWords,
        FeaturesDistortedView,
        FeaturesMendenhall,
        FeaturesSentenceLength,
        FeaturesPOST,
        FeatureSetReductor,
        FeaturesDEP,
        FeaturesPunctuation,
        HstackFeatureSet,
        FeaturesFrequentWords
)

from learner import ClassifierRange

import warnings

from src.feature_extraction.features import FeaturesCharNGram

warnings.filterwarnings("ignore")

import pandas as pd
import os

# ----------------------------------------------
# Save results
# ----------------------------------------------

class SaveResults:
    def __init__(self, config: "ModelConfig", mode: str = "inference"):
        """
        Handles saving results for both inference and leave-one-out (LOO) modes.

        - In 'inference' mode: saves basic predictions (no metrics)
        - In 'loo' mode: saves predictions + evaluation metrics (accuracy, f1, confusion matrix)
        """
        if mode not in ("inference", "loo"):
            raise ValueError("mode must be 'inference' or 'loo'")

        self.config = config
        self.mode = mode

        # Columns depend on the mode
        if mode == "loo":
            columns = [
                "best_params", "booktitle", "author", "predictedauthor", "posterior_prob", "type",
                "accuracy", "TN", "FP", "FN", "TP"
            ]
        elif mode == "inference":  # inference mode
            columns = [
                "best_params", "booktitle", "author", "predictedauthor", "posterior_prob", "best_score", "type"
            ]

        self.df = pd.DataFrame(columns=columns)

    def add_result(
            self,
            best_params,
            booktitle,
            author,
            predictedauthor,
            posterior_prob,
            type,
            accuracy=None,
            best_score=None,
            TN=None,
            FP=None,
            FN=None,
            TP=None,
    ):
        new_row = {
            "best_params": best_params,
            "booktitle": booktitle,
            "author": author,
            "predictedauthor": predictedauthor,
            "posterior_prob": posterior_prob,
            "type": type,
        }

        if self.mode == "loo":
            new_row.update({
                "accuracy": accuracy,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "TP": TP,
            })

        elif self.mode == "inference":
            new_row["best_score"] = best_score

        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    def save(self):
        """Save the accumulated results to the appropriate CSV file."""
        result_path = (
            self.config.results_inference
            if self.mode == "inference"
            else self.config.results_loo
        )
        self.df.to_csv(result_path, index=False)

# ----------------------------------------------
# Evaluate model
# ----------------------------------------------

class ModelEvaluator:

    def __init__(self, config: "ModelConfig"):
        """
        evaluator for binary model, confusion matrix not yet implemented
        """
        self.config = config

    def evaluate (self, y:list, y_pred:list):

        pos_label = self.config.positive_author
        neg_label = f"Not{pos_label}"
        labels = neg_label, pos_label

        acc = accuracy_score(y, y_pred)

        cm = confusion_matrix(y, y_pred, labels=labels)
        tn, fp, fn, tp = cm.ravel()

        return acc, tn, fp, fn, tp


# ----------------------------------------------
# Model
# ----------------------------------------------
class AuthorshipVerification:
    """Main class for authorship verification system"""

    def __init__(self, config: "ModelConfig", nlp: spacy.Language):
        self.config = config
        self.nlp = nlp
        self.cls = None
        self.best_params = None
        self.best_score = None

    def feature_extraction_fit(self, processed_docs: List[spacy.tokens.Doc], y: List[str]):

        spanish_function_words = get_spanish_function_words()

        vectorizers_dict = {
            'feat_funct_words': FeaturesFunctionWords(
                function_words=spanish_function_words,
                ngram_range=(1, 1)
            ),
            'feat_post': FeatureSetReductor(
                 FeaturesPOST(n=(1, 3)),
                 max_features=self.config.max_features
            ),
            'feat_mendenhall': FeaturesMendenhall(upto=27),
            'feat_sentlength': FeaturesSentenceLength(),
            'feat_dvex': FeatureSetReductor(
                 FeaturesDistortedView(method="DVEX",
                                      function_words=spanish_function_words),
                max_features=self.config.max_features
            ),
            'feat_punct': FeaturesPunctuation(),
            'feat_dep': FeatureSetReductor(
                 FeaturesDEP(n=(2, 3), use_words=True),
                 max_features=self.config.max_features
            ),
            'feat_char': FeatureSetReductor(
                FeaturesCharNGram(n=(1,3))
            ),
            "feat_k_freq_words": FeaturesFrequentWords()
        }

        names, vectorizers = list(zip(*vectorizers_dict.items()))

        self.hstacker = HstackFeatureSet(*vectorizers, verbose=True)
        X = self.hstacker.fit_transform(processed_docs, y)
        y = np.asarray(y)
        slices = self.hstacker.get_feature_slices()
        slices_dict = {name_: slice_ for name_,slice_ in zip(names, slices)}
        return X, y, slices_dict

    def feature_extraction_transform(self, processed_docs: List[spacy.tokens.Doc]):
        return self.hstacker.transform(processed_docs)

    def prepare_X_y(self, train_documents: List[Book]):
        texts = []
        labels = []
        groups = []
        for i, book in enumerate(train_documents):
            label = book.author
            texts.append(book.processed)
            labels.append(label)
            groups.append(i)
            for segment in book.segmented:
                texts.append(segment)
                labels.append(label)
                groups.append(i)
        X, y, slices = self.feature_extraction_fit(texts, labels)
        return X, y, slices, groups

    def prepare_classifier(self):
        classifier_type = getattr(self.config, "classifier_type", "lr")
        print(f"Building classifier: {classifier_type}\n")

        if classifier_type == "lr":
            base_estimator = LogisticRegression(random_state=self.config.random_state, n_jobs=self.config.n_jobs)
        elif classifier_type == "svm":
            base_estimator = LinearSVC(random_state=self.config.random_state)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")

        cls_range = ClassifierRange(base_cls=base_estimator)
        return cls_range


    def fit(self, train_documents: List[Book], save_hyper_path:str=None):

        X, y, slices, groups = self.prepare_X_y(train_documents)

        # model selection
        cls_range = self.prepare_classifier()

        mod_selection = GridSearchCV(
            estimator=cls_range,
            param_grid={
                'C': np.logspace(-4, 4, 9),
                'class_weight': [None, 'balanced'],
                'feat_funct_words': [None, slices['feat_funct_words']],
                'feat_post': [None, slices['feat_post']],
                'feat_mendenhall': [None, slices['feat_mendenhall']],
                'feat_sentlength': [None, slices['feat_sentlength']],
                'feat_dvex': [None, slices['feat_dvex']],
                'feat_punct': [None, slices['feat_punct']],
                'feat_dep': [None, slices['feat_dep']],
                'feat_char': [None, slices['feat_char']],
                'feat_k_freq_words': [None, slices['feat_k_freq_words']],
            },
            cv=LeaveOneGroupOut(),
            refit=False,
            verbose=1,
            scoring=make_scorer(f1_score, pos_label=self.config.positive_author, zero_division=1.0),
            n_jobs=-1
        )
        print(set([(yi,gi) for yi, gi in zip(y,groups)]))
        mod_selection.fit(X, y, groups=groups)

        self.best_params = mod_selection.best_params_
        best_params = mod_selection.best_params_
        self.best_score = mod_selection.best_score_
        print('best params:', mod_selection.best_params_)
        print('best score:', mod_selection.best_score_)

        if save_hyper_path is not None:
            parent = Path(save_hyper_path).parent
            os.makedirs(parent, exist_ok=True)
            pickle.dump(self.best_params, open(save_hyper_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\nBuilding classifier: classifier calibration ({cls_range.__class__.__name__})\n")
        self.create_classifier(X, y, cls_range, self.best_params)


    def create_classifier(self, X, y, cls:BaseEstimator, hyperparams:dict):
        cls_optim = clone(cls)
        cls_optim.set_params(**hyperparams)

        self.cls = CalibratedClassifierCV(
            cls_optim,
            cv=10,
            #method="isotonic",
            method="sigmoid",
            n_jobs=-1,
        )

        self.cls.fit(X, y)
        print('[done]')
        return self

    def fit_with_hyperparams(self, train_documents: List[Book], hyperparams: dict):
        X, y, slices, groups = self.prepare_X_y(train_documents)

        def assert_coherent_slices(slices, hyperparams):
            for feat, slice in hyperparams.items():
                if slice is not None:
                    assert slice == slices[feat], f'wrong slices for feat {feat}'

        assert_coherent_slices(slices, hyperparams)
        cls_range = self.prepare_classifier()
        print(f"\nBuilding classifier: classifier calibration ({cls_range.__class__.__name__})\n")
        self.create_classifier(X, y, cls_range, hyperparams)


    def leave_one_out(self, train_documents: List[Book]):
        assert self.cls is not None, 'leave_one_out called before fit!'

        texts, labels, groups, titles, segment_flags = [], [], [], [], []
        segment_prediction= {}

        saver = SaveResults(self.config, mode="loo") if self.config.results_loo else None
        evaluator = ModelEvaluator(self.config)

        for i, book in enumerate(train_documents):
            label = book.author
            texts.append(book.processed)
            labels.append(label)
            groups.append(i)
            titles.append(book.title)
            segment_flags.append("full_book")

            for segment in book.segmented:
                texts.append(segment)
                labels.append(label)
                groups.append(i)
                titles.append(book.title)
                segment_flags.append("segment")

        X, y, _ = self.feature_extraction_fit(texts, labels)

        loo = LeaveOneGroupOut()

        for train_index, test_index in loo.split(X, y, groups):
            Xtr, ytr = X[train_index], y[train_index]
            Xte, yte = X[test_index], y[test_index]

            cls_clone = clone(self.cls)
            cls_clone.fit(Xtr, ytr)

            predictions = cls_clone.predict(Xte)
            posteriors = cls_clone.predict_proba(Xte)

            for idx in range(len(test_index)):
                book_prediction = predictions[idx]
                book_label = yte[idx]
                book_title = titles[test_index[idx]]
                pred_idx = self.index_of_author(book_prediction)
                posterior_prob = posteriors[idx][pred_idx]
                flag = segment_flags[test_index[idx]]

                acc = tn = fp = fn = tp = None

                if flag == "segment":
                    if book_title not in segment_prediction:
                        segment_prediction[book_title] = {"y": [], "y_pred": []}

                    segment_prediction[book_title]["y"].append(book_label)
                    segment_prediction[book_title]["y_pred"].append(book_prediction)

                    acc, tn, fp, fn, tp = evaluator.evaluate([book_label], [book_prediction])

                elif flag == "full_book":
                    for book in segment_prediction:
                        y_true = segment_prediction[book]["y"]
                        y_pred = segment_prediction[book]["y_pred"]
                        acc, tn, fp, fn, tp = evaluator.evaluate(y_true, y_pred)


                if saver is not None:
                    saver.add_result(
                        best_params = self.best_params,
                        booktitle=book_title,
                        author=book_label,
                        predictedauthor=book_prediction,
                        posterior_prob=posterior_prob,
                        type=flag,
                        accuracy=acc,
                        TN=tn,
                        FP=fp,
                        FN=fn,
                        TP=tp,
                    )

        if saver is not None:
            saver.save()


    def predict(self, test_corpus: List[Book], return_posteriors=False):

        texts = [book.processed for book in test_corpus]
        labels = [book.author for book in test_corpus]
        titles = [book.title for book in test_corpus]

        X = self.feature_extraction_transform(texts)

        y_predicted = self.cls.predict(X)

        if return_posteriors:
            posteriors = self.cls.predict_proba(X)

            if self.config.results_inference:
                saver = SaveResults(self.config, mode="inference")
                for title, author, pred_author, posterior in zip(
                        titles, labels, y_predicted, posteriors
                ):
                    pred_idx = self.index_of_author(pred_author)
                    saver.add_result(
                        best_params= self.best_params,
                        best_score = self.best_score,
                        booktitle=title,
                        author=author,
                        predictedauthor=pred_author,
                        posterior_prob=posterior[self.config.positive_author],
                        type="full book"
                    )
                saver.save()
            else: None

            return y_predicted, posteriors

        else:
            return y_predicted

    @property
    def classes(self):
        return self.cls.classes_

    def index_of_author(self, author):
        return self.classes.tolist().index(author)
