from typing import List, Tuple, Dict, Optional, Any, Union
import numpy as np
import spacy
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedGroupKFold,
                                     GridSearchCV,
                                     LeaveOneGroupOut)
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
)

import warnings
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
                "booktitle", "author", "predictedauthor", "posterior_prob", "type",
                "accuracy", "f1", "TP", "TN", "FP", "FN"
            ]
        else:  # inference mode
            columns = [
                "booktitle", "author", "predictedauthor", "posterior_prob", "type"
            ]

        self.df = pd.DataFrame(columns=columns)

    def add_result(
        self,
        booktitle,
        author,
        predictedauthor,
        posterior_prob,
        type,
        accuracy=None,
        f1=None,
        TP=None,
        TN=None,
        FP = None,
        FN=None,
    ):
        """
        Add a single prediction result (with optional metrics).
        """
        new_row = {
            "booktitle": booktitle,
            "author": author,
            "predictedauthor": predictedauthor,
            "posterior_prob": posterior_prob,
            "type": type,
        }

        # Add metrics only in LOO mode)
        if self.mode == "loo":
            new_row.update({
                "accuracy": accuracy,
                "f1": f1,
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
            })

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

class   ModelEvaluator:

    def __init__(self, config: "ModelConfig"):
        """
        evaluator for binary model, confusion matrix not yet implemented
        """
        self.config = config

    def evaluate (self, y:list, y_pred:list):

        pos_label = self.config.positive_author
        neg_label = f"Not{pos_label}"
        labels = pos_label, neg_label

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='binary', pos_label=pos_label, zero_division=1)

        cm = confusion_matrix(y, y_pred, labels=labels)
        tn, fp, fn, tp = cm.ravel()

        return acc, f1, tp, tn, fp, fn


# ----------------------------------------------
# Model
# ----------------------------------------------
class AuthorshipVerification:
    """Main class for authorship verification system"""

    def __init__(self, config: "ModelConfig", nlp: spacy.Language):
        self.config = config
        self.nlp = nlp
        self.cls = None

    def feature_extraction_fit(self, processed_docs: List[spacy.tokens.Doc], y: List[str]):

        spanish_function_words = get_spanish_function_words()

        vectorizers = [
            FeaturesFunctionWords(
                function_words=spanish_function_words,
                ngram_range=(1, 1)
            ),
            FeatureSetReductor(
                FeaturesPOST(n=(1, 3)),
                max_features=self.config.max_features
            ),
            FeaturesMendenhall(upto=27),
            FeaturesSentenceLength(),
            FeatureSetReductor(
                FeaturesDistortedView(method="DVEX",
                                      function_words=spanish_function_words),
                max_features=self.config.max_features
            ),
            FeaturesPunctuation(),
            FeatureSetReductor(
                FeaturesDEP(n=(2, 3), use_words=True),
                max_features=self.config.max_features
            ),
        ]

        self.hstacker = HstackFeatureSet(*vectorizers, verbose=True)
        X = self.hstacker.fit_transform(processed_docs, y)
        y = np.asarray(y)
        return X, y

    def feature_extraction_transform(self, processed_docs: List[spacy.tokens.Doc]):
        return self.hstacker.transform(processed_docs)

    def fit(self, train_documents: List[Book]):

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

        X, y = self.feature_extraction_fit(texts, labels)

        # model selection
        print(f"Building classifier: model selection\n")
        mod_selection = GridSearchCV(
            estimator=LogisticRegression(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            ),
            param_grid={
                'C': np.logspace(-4, 4, 9),
                'class_weight': [None, 'balanced']
            },
            cv=LeaveOneGroupOut(),
            refit=False,
            verbose=1,
            scoring=make_scorer(f1_score, pos_label=self.config.positive_author, zero_division=1.0),
            n_jobs=-1
        )
        print(set([(yi,gi) for yi, gi in zip(y,groups)]))
        mod_selection.fit(X, y, groups=groups)
        # cv_results = pd.DataFrame(mod_selection.cv_results_)


        best_params = mod_selection.best_params_
        print('best params:', mod_selection.best_params_)
        print('best score:', mod_selection.best_score_)

        print(f"Building classifier: classifier calibration\n")
        self.cls = CalibratedClassifierCV(
            LogisticRegression(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                **best_params
            ),
            cv=10,
            # method='isotonic',
            method='sigmoid',
            n_jobs=-1
        )
        self.cls.fit(X, y)

        print('[done]')

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

        X, y = self.feature_extraction_fit(texts, labels)

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

                acc = f1 = tp = tn = fp = fn = None

                if flag == "segment":
                    if book_title not in segment_prediction:
                        segment_prediction[book_title] = {"y": [], "y_pred": []}

                    segment_prediction[book_title]["y"].append(book_label)
                    segment_prediction[book_title]["y_pred"].append(book_prediction)

                    acc, f1, tp, tn, fp, fn = evaluator.evaluate([book_label], [book_prediction])

                elif flag == "full_book":
                    for book in segment_prediction:
                        y = segment_prediction[book]["y"]
                        y_pred = segment_prediction[book]["y_pred"]
                        acc, f1, tp, tn, fp, fn = evaluator.evaluate(y, y_pred)



                if saver is not None:
                    saver.add_result(
                        booktitle=book_title,
                        author=book_label,
                        predictedauthor=book_prediction,
                        posterior_prob=posterior_prob,
                        type=flag,
                        accuracy=acc,
                        f1=f1,
                        TP=tp,
                        TN=tn,
                        FP=fp,
                        FN=fn,
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
                        booktitle=title,
                        author=author,
                        predictedauthor=pred_author,
                        posterior_prob=posterior[pred_idx],
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

        filenames = [book.path.name for book in test_corpus]
        processed_documents = [book.processed for book in test_corpus]

        (X_stacked, y, filenames, feature_sets_idxs, *_) = self.feature_extraction_fit(
        processed_documents, y, filenames)

        y_pred = clf.predict(X_stacked)

        results = {
            'filenames': filenames,
            'label': y,
            'prediction': y_pred,
            'feature_sets_idxs': feature_sets_idxs
        }
        return results
