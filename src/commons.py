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
        if mode is "inference", the file is saved in config.results_inference,
        if mode is "loo", the file is saved in config.results_loo
        mode is specified in the class AuthorshipVerification
        """
        if mode not in ("inference", "loo"):
            raise ValueError("mode must be 'inference' or 'loo'")

        self.config = config
        self.mode = mode
        self.df = pd.DataFrame(columns=[
            "booktitle", "author", "predictedauthor", "posterior_prob", "type"
        ])

    def add_result(self, booktitle, author, predictedauthor, posterior_prob, type):
        new_row = {
            "booktitle": booktitle,
            "author": author,
            "predictedauthor": predictedauthor,
            "posterior_prob": posterior_prob,
            "type": type
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    def save(self):
        if self.mode == "inference":
            result_path = self.config.results_inference
        else:
            result_path = self.config.results_loo

        self.df.to_csv(result_path, index=False)


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


        # feature_sets = []
        # feature_sets_orig = []
        # orig_filenames = filenames.copy()

        # for vectorizer in vectorizers:
        #     reductor = FeatureSetReductor(vectorizer, keep_ratio=self.config.keep_ratio)
        #
        #     # print('\nProcessing set')
        #     features_set= reductor.fit_transform(processed_docs, y)
        #
        #     if self.config.oversample:
        #         feature_sets_orig.append(features_set)
        #         orig_y = y.copy()
        #
        #         (
        #             features_set,
        #             y_oversampled,
        #             features_set,
        #             y_oversampled,
        #             groups,
        #         ) = reductor.oversample_DRO(
        #             Xtr=features_set,
        #             ytr=y,
        #             groups=orig_filenames,
        #             rebalance_ratio=self.config.rebalance_ratio,
        #         )
        #         feature_sets.append(features_set)
        #     else:
        #         feature_sets.append(features_set)

        # orig_feature_sets_idxs = self._compute_feature_set_idx(
        #     vectorizers, feature_sets_orig
        # )
        # feature_sets_idxs = self._compute_feature_set_idx(vectorizers, feature_sets)

        # print(f'Feature sets computed: {len(feature_sets_dev)}')
        # print('\nStacking feature vectors')

        # if feature_sets_orig:
        #     X_stacked_orig = hstacker._hstack(feature_sets_orig)
            # print(f'X_dev_stacked_orig shape: {X_dev_stacked_orig.shape}')
            # print(f'X_test_stacked_orig shape: {X_test_stacked_orig.shape}')

        # X_stacked = hstacker._hstack(feature_sets)

        # print(f'X_dev_stacked shape: {X_dev_stacked.shape}')
        # print(f'X_test_stacked shape: {X_test_stacked.shape}')

        # y_final = y_oversampled if self.config.oversample else y

        #print("Feature vectors extracted.")
        #print(f"Vector document final shape: {X_dev_stacked.shape}")
        #print(f"X_dev_stacked: {X_dev_stacked.shape[0]}")
        #print(f"y_dev: {len(y_dev_final)}")
        #print(f"groups_dev: {len(groups_dev)}")
        #print(f"groups_dev_orig: {len(orig_groups_dev)}")

        # if self.config.oversample:
        #     return (X_stacked, y_final, filenames, feature_sets_idxs,
        #         orig_feature_sets_idxs, X_stacked_orig, orig_y, orig_filenames)
        # else:
        #     return (X_stacked, y, filenames, feature_sets_idxs, None, None, None, None)

    def _compute_feature_set_idx(self, vectorizers, feature_sets_dev):
        """Helper method to compute feature set indices"""
        start_idx = 0
        end_idx = 0
        feature_sets_idxs = {}

        for vect, fset in zip(vectorizers, feature_sets_dev):
            if isinstance(fset, list):
                fset = np.array(fset)

            if len(fset.shape) == 1:
                fset = fset.reshape(-1, 1)

            feature_shape = fset.shape[1]
            end_idx += feature_shape
            feature_sets_idxs[vect] = (start_idx, end_idx)
            start_idx = end_idx

        return feature_sets_idxs

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

        texts = []
        labels = []
        groups = []
        titles = []
        segment_flags = []

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
        saver = SaveResults(self.config, mode="loo") if self.config.results_loo else None

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

                if saver is not None:
                    saver.add_result(
                        booktitle=book_title,
                        author=book_label,
                        predictedauthor=book_prediction,
                        posterior_prob=posterior_prob,
                        type=flag
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
