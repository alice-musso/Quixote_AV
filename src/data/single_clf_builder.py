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
    FeaturesFunctionWords, 
    FeaturesPunctuation,
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

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier

import pickle

import warnings
warnings.filterwarnings('ignore') 

import time

start_time_global = time.time() 

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
        print('\nPositive test available\n')
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

def extract_features(X_dev, X_test, y_dev, X_test_frag, function_ws):
    function_words_extractor = FeaturesFunctionWords(language='spanish', function_words=function_ws)
    mendenhall_extractor = FeaturesMendenhall(upto=20)
    words_masker = FeaturesDVEX(function_words=function_ws)
    sentence_len_extractor = FeaturesSentenceLength()
    POS_extractor = FeaturesPOST(language='spanish')
    DEP_extractor = FeaturesDEP()
    punct_extractior = FeaturesPunctuation(punctuation=punctuation)

    vectorizers = [function_words_extractor, mendenhall_extractor, words_masker, sentence_len_extractor, POS_extractor, punct_extractior, DEP_extractor]
    hstacker = HstackFeatureSet(vectorizers)

    pounct_reductor = FeatureSetReductor(punct_extractior)
    punct_red = pounct_reductor.fit_transform(X_dev, authors=y_dev)
    punct_red_test = pounct_reductor.transform(X_test)

    fw_reductor = FeatureSetReductor(function_words_extractor) 
    fw_red = fw_reductor.fit_transform(X_dev, authors=y_dev)
    fw_red_test = fw_reductor.transform(X_test)

    sl_reductor = FeatureSetReductor(sentence_len_extractor) 
    sl_red = sl_reductor.fit_transform(documents=X_dev, authors=y_dev)
    sl_red_test = sl_reductor.transform(X_test)

    me_reductor = FeatureSetReductor(mendenhall_extractor)
    me_red = me_reductor.fit_transform(X_dev, authors=y_dev)
    me_red_test = me_reductor.transform(X_test)

    wm_reductor = FeatureSetReductor(words_masker)
    wm_red = wm_reductor.fit_transform(X_dev, authors=y_dev)
    wm_red_test = wm_reductor.transform(X_test)

    pos_reductor = FeatureSetReductor(POS_extractor)
    pos_red = pos_reductor.fit_transform(X_dev, authors=y_dev)
    pos_red_test = pos_reductor.transform(X_test)

    dep_reductor = FeatureSetReductor(DEP_extractor)
    dep_red = dep_reductor.fit_transform(X_dev, authors=y_dev)
    dep_red_test = dep_reductor.transform(X_test)


    punct_red_test_fragments = pounct_reductor.transform(X_test_frag)
    fw_red_test_fragments = fw_reductor.transform(X_test_frag)
    sl_red_test_fragments = sl_reductor.transform(X_test_frag)
    me_red_test_fragments = me_reductor.transform(X_test_frag)
    wm_red_test_fragments = wm_reductor.transform(X_test_frag)
    pos_red_test_fragments = pos_reductor.transform(X_test_frag)
    dep_red_test_fragments = dep_reductor.transform(X_test_frag)

    feature_sets_dev = [fw_red, me_red, wm_red, pos_red, sl_red, punct_red, dep_red]
    feature_sets_test = [fw_red_test, me_red_test, wm_red_test, pos_red_test, sl_red_test, punct_red_test, dep_red_test]
    feature_sets_test_frag = [fw_red_test_fragments, me_red_test_fragments, wm_red_test_fragments, pos_red_test_fragments, sl_red_test_fragments, punct_red_test_fragments, dep_red_test_fragments]

    # hstacked_features = hstacker._hstack([fw_red, me_red, wm_red, pos_red, sl_red, punct_red, dep_red])
    # hstacked_features_test = hstacker._hstack([fw_red_test, me_red_test, wm_red_test, pos_red_test, sl_red_test, punct_red_test, dep_red_test])
    # hstacked_features_test_frag = hstacker._hstack([fw_red_test_fragments, me_red_test_fragments, wm_red_test_fragments, pos_red_test_fragments, sl_red_test_fragments, punct_red_test_fragments, dep_red_test_fragments])

    X_dev_stacked = hstacker._hstack(feature_sets_dev)
    X_test_stacked = hstacker._hstack(feature_sets_test)
    X_test_stacked_frag = hstacker._hstack(feature_sets_test_frag)
    return X_dev_stacked, X_test_stacked, X_test_stacked_frag


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


# main

def build_single_clf():
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
    
    for author in np.unique(authors):
        
        start_time = time.time()
        print()
        print('Single classifier building for author:', author)
        X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, groups_test, groups_test_frag = split_n_segment(documents=documents, authors=authors, target=author)

        X_dev_stacked, X_test_stacked, X_test_stacked_frag = extract_features(X_dev=X_dev, y_dev=y_dev, X_test=X_test, X_test_frag=X_test_frag, function_ws=fuction_ws)

    
        clf = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
        f1_binary = make_scorer(f1_score, pos_label=1)

        param_grid= {'C': np.logspace(-4,4,9), 
                    'random_state': [42]}

        grid = GridSearchCV(
                    clf,
                    param_grid=param_grid,
                    cv=5,
                    n_jobs=5,
                    scoring=f1_binary,
                    verbose=True)

        grid.fit(X_dev_stacked, y_dev, groups=groups_dev)
        print('Model fitted. Best params:')
        print(grid.best_params_)
        print()

        print('Saving model')
        clf = grid.best_estimator_
        save_model(clf,'Single_clf_', author, folder='SingleClf_models')
        print('Model saved.')
        print()
        print('Computing res...')
        print('On whole docs')
        y_pred = get_scores(grid.best_estimator_, X_test_stacked, y_test)
        print()
        print('On fragmented docs:')
        y_pred_frag = get_scores(grid.best_estimator_, X_test_stacked_frag, y_test_frag)
        print()
        print('Model building for author', author, 'completed')
        print('Time spent for model building for author',author,':', time.time() - start_time)

    
    print('Models built. Termination time', time.localtime())
    print("Total time spent", time.time() - start_time_global)


build_single_clf() 
# manca run da pasamonte in poi 