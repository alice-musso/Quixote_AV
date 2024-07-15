import os

from tqdm import tqdm

from data_loader import load_spanish_corpus

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import re

from splitting__ import Segmentation

from helpers__ import get_function_words

from string import punctuation

from features import ( 
    FeaturesPunctuation,
    FeaturesFunctionWords, 
    FeaturesDVEX, 
    FeaturesMendenhall, 
    FeaturesSentenceLength, 
    FeaturesPOST, 
    FeaturesDEP,
    FeatureSetReductor,
    HstackFeatureSet
)

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    make_scorer,
)

#from scikitplot.metrics import plot_roc

from sklearn.ensemble import StackingClassifier

import pickle

import warnings
warnings.filterwarnings('ignore') 

import time

start_time_global = time.time() 

# evaluation metric
f1_binary = make_scorer(f1_score, pos_label=1)

# support classes


class Fun_StackingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, vectorizers, C_svm=1, kernel_svm='rbf', gamma_svm='scale', degree_svm=3, normalize=False, use_cv=False):
        self.vectorizers = vectorizers
        self.normalize = normalize
        self.use_cv = use_cv
        
        self.C_svm = C_svm
        self.kernel_svm = kernel_svm
        self.gamma_svm=gamma_svm
        self.degree_svm= degree_svm

        
    def fit(self, X, y):

        base_clfs = [SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced') for _ in self.vectorizers]
        systems = [Pipeline([
        ('feat_extractor', vectorizer),
        ('classifier', base_clf)
        ]) 
        for vectorizer, base_clf in zip(self.vectorizers, base_clfs)]

        for system in systems:
            system.fit(X,y)

        estimators = [(f'model_{i+1}', system) for i,system in enumerate(systems)]


        stack = StackingClassifier(
                estimators=estimators,
                final_estimator=Pipeline([
                                          ('Normalizer', StandardScaler() if self.normalize else None),
                                          ('meta_classifier', GridSearchCV(
                                              estimator=SVC(C=self.C_svm, kernel=self.kernel_svm, gamma=self.gamma_svm, degree=self.degree_svm, class_weight='balanced'),
                                              param_grid={
                                                    'C': np.logspace(-4,4,9),
                                                    'kernel':['rbf', 'poly'],#aggiungere poly?
                                                    'gamma': ['scale', 'auto'],
                                                    #'degree_svm': [3],
                                                },
                                                n_jobs=5,
                                                scoring=f1_binary,
                                                cv=5,
                                                verbose=True
                                          ))
                                           
                ]),
                cv=5 if self.use_cv else 'prefit',  
                stack_method='predict_proba',
                n_jobs=3,
                verbose=True
            )

        stack.fit(X, y)

        self.classes_ = stack.classes_

        self.stack = stack

        self.meta_best_params = stack.final_estimator_.named_steps['meta_classifier'].best_params_

        return self
    
    def predict(self, X):
        return self.stack.predict(X)

# support functions

def split(X, y):

    if np.sum(y) < 2:
        print()
        print('One doc only author')
        print()

        positive_doc_idx = y.index(1)
        pos_X = X[positive_doc_idx]
        pos_X = np.expand_dims(pos_X, axis=0)
        pos_y = y[positive_doc_idx]

        neg_X = np.delete(X, positive_doc_idx)
        neg_y = np.delete(y, positive_doc_idx)
        
        X_dev_neg, X_test, y_dev_neg, y_test = train_test_split(
            neg_X, neg_y, test_size=0.3, random_state=42
        )
        X_dev = np.concatenate((pos_X, X_dev_neg), axis=0)
        y_dev = np.concatenate(([pos_y], y_dev_neg), axis=0)

        X_dev = list(X_dev)
        y_dev = list(y_dev)
        X_test = list(X_test)
        y_test = list(y_test)

        
    else:
        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
    return X_dev, X_test, y_dev, y_test 



def split_n_segment(documents, target, authors, min_tokens=500):

    X = documents
    y = [1 if author.rstrip() == target else 0 for author in authors]

    X_dev, X_test, y_dev, y_test = split(X, y)

    segmentator_dev = Segmentation(split_policy='by_sentence', tokens_per_fragment=min_tokens)
    splitted_docs_dev = segmentator_dev.fit_transform(documents=X_dev, authors=y_dev)
    groups_dev = segmentator_dev.groups

    segmentator_test = Segmentation(split_policy='by_sentence', tokens_per_fragment=min_tokens)
    splitted_docs_test = segmentator_test.transform(documents=X_test, authors=y_test)
    groups_test = segmentator_test.groups

    X_dev = splitted_docs_dev[0]
    y_dev = splitted_docs_dev[1]
    groups_dev = segmentator_dev.groups

    X_test = splitted_docs_test[0][:10] # whole_docs_test
    y_test = splitted_docs_test[1][:10] # whole_docs_y_test

    X_test_frag = splitted_docs_test[0][10:]
    y_test_frag = splitted_docs_test[1][10:]
    groups_test_frag = groups_test[10:] 

    return X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, groups_test, groups_test_frag

def save_model(model, filename, target, folder):
    path = f'/home/martinaleo/.ssh/authorship/src/data/{folder}'
    os.chdir(path)
    filename = f"{target}-{filename}.pickle"
    # save model
    pickle.dump(model, open(filename, "wb"))


def load_model(filename, target, folder):
    path = f'/home/martinaleo/.ssh/authorship/src/data/{folder}'
    os.chdir(path)
    #os.chdir('/home/martinaleo/.ssh/authorship/src/data/FunTAT_normalize_false_models')
    filename = f"{target}-{filename}.pickle"
    return pickle.load(open(filename, "rb"))


def get_scores(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=1.0)
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=1.0)
    cf = confusion_matrix(y_test, y_pred).ravel() #(tn, fp, fn, tp)

    print('Precision:', precision)
    print('Recall:', recall)
    print('Accuracy:', acc)
    print('F1:', f1)
    print()
    print(classification_report(y_test, y_pred, zero_division=1.0))
    print()
    print('Confusion matrix: (tn, fp, fn, tp)\n', cf)

    if np.sum(y_test)>0:
        print()
        print('Confusion Matrix:')
        print('TP:', cf[3], '\t|  FP', cf[1])
        print('FN:', cf[2], '\t|  TN', cf[0])

    return y_pred


def plot_res(y_test, y_pred, target):
    y_labels = (target, 'Other')
    cf = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cf, annot=True, fmt='.2f', cmap="Blues", xticklabels=y_labels, yticklabels=y_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    df_res = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    sns.countplot(x="variable", hue="value", data=pd.melt(df_res))
    plt.title('Actual distribution vs predicted', fontsize=14)

    plt.legend(loc='best', bbox_to_anchor=(1, 0.9), fancybox=True, shadow=True)
    plt.show()


def build_fun_model(documents, authors, vectorizers, normalize=False, use_kfcv=False):

    for author in np.unique(authors):
        start_time = time.time()
        model_str = 'KFCV' if use_kfcv else 'TAT'
        print(f'{model_str}-model building for author:', author, '(with z-scoring)' if normalize else '(no normalization)')
        print()
        X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, groups_test, groups_test_frag = split_n_segment(documents=documents, authors=authors, target=author)

        grid = GridSearchCV(
        estimator=Fun_StackingClassifier(vectorizers=vectorizers, normalize=normalize, use_cv=use_kfcv),
        param_grid={},
        n_jobs=3,
        scoring=f1_binary,
        cv=5,
        verbose=True
        )

        print()
        print('Fitting model...')
        print()
        grid.fit(X_dev, y_dev, groups=groups_dev)
        print('Model fitted. Best meta params:')
        print(grid.best_estimator_.meta_best_params)
        print()
        print('Saving model')
        clf_fan = grid.best_estimator_
        if normalize:
            save_model(clf_fan, f'Fun{model_str.capitalize()}clf_norm_True', author, folder=f'Fun{model_str}_normalize_true_models')
        else:
            save_model(clf_fan, f'Fun_{model_str.capitalize()}_clf_norm_False', author, folder=f'Fun{model_str}_normalize_false_models')
        print('Model saved.')
        print()
        print('Computing res...')
        print('On whole docs')
        y_pred = get_scores(grid.best_estimator_, X_test, y_test)
        print()
        print('On fragmented docs:')
        y_pred_frag = get_scores(grid.best_estimator_, X_test_frag, y_test_frag)
        print()
        print('Model building for author', author, 'completed')
        print('Time spent for model building for author',author,':', time.time() - start_time)
        print()

    print()
    print('All models built.')


def build_model(model_type='tat', normalize=False):
    print('Start time:', time.localtime())

    # data loading
    documents, authors, filenames = load_spanish_corpus(path='/home/martinaleo/authorship/src/data/Corpus')

    # removing target doc
    documents = documents[:3] + documents[4:]
    authors= authors[:3] + authors[4:]
    filenames= filenames[:3] + filenames[4:]

    # cleaning
    documents = [document.lower() for document in documents]
    documents = [(re.sub(r'\[(?:(?!\[|\]).)*\]', '', document)) for document in documents] 
    # elimina le parti di testo delimitate da [] e che non contengono al loro interno ][
    authors = [author.rstrip() for author in authors]

    # vectorizers for feature extraction
    fuction_ws = get_function_words(lang='spanish')

    punct_extractior = FeaturesPunctuation(punctuation=punctuation)
    function_words_extractor = FeaturesFunctionWords(language='spanish', function_words=fuction_ws)
    mendenhall_extractor = FeaturesMendenhall(upto=20)
    words_masker = FeaturesDVEX(function_words=fuction_ws)
    sentence_len_extractor = FeaturesSentenceLength()
    POS_extractor = FeaturesPOST(language='spanish')
    DEP_extractor = FeaturesDEP()

    vectorizers = [punct_extractior, function_words_extractor, mendenhall_extractor, words_masker, sentence_len_extractor, POS_extractor, DEP_extractor]

    # model building
    if model_type.lower() == 'tat':
        build_fun_model(documents, authors, vectorizers, normalize=normalize, use_kfcv=False)

    elif model_type.lower() == 'kfcv':
        build_fun_model(documents, authors, vectorizers, normalize=normalize, use_kfcv=True)
    
    print('Models built. Termination time', time.localtime())
    print("Total time spent", time.time() - start_time_global)

# build_model(model_type='tat', normalize=False) # runnato con features sintattiche (2,3)
# print('--------------------------------------------------------------------------------')
# build_model(model_type='kfcv', normalize=False)# runnato
# print('--------------------------------------------------------------------------------')
# build_model(model_type='tat', normalize=True) # runnato con features sintattiche (2,3)
# print('--------------------------------------------------------------------------------')
# build_model(model_type='kfcv', normalize=True) # runnato con features sintattiche (2,3) 