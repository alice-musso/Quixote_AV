from tqdm.notebook import tqdm
import os
import csv

from data_loader import load_corpus

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


import re

from nltk import download
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import spacy
import es_core_news_sm

from splitting__ import Segmentation

from helpers__ import get_function_words

from features import ( 
    FeaturesFunctionWords, 
    FeaturesDVEX, 
    FeaturesMendenhall, 
    FeaturesSentenceLength, 
    FeaturesPOST, 
    FeatureSetReductor,
    FeaturesDEP,
    FeaturesPunctuation,
    HstackFeatureSet,
    #FeaturesCharNGram
)


from sklearn.feature_selection import SelectKBest, chi2

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    RandomizedSearchCV,
    GridSearchCV,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    precision_recall_fscore_support
)

from scikitplot.metrics import plot_roc

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier

from imblearn.over_sampling import SMOTE

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap


def load_dataset(remove_test=False):
    path = '/home/martinaleo/Quaestio_AV/authorship/src/data/Quaestio-corpus'
    documents, authors, filenames = load_corpus(path=path)

    #code to remove pattern
    return documents, authors, filenames


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
        print('\nAuthor with multiple documents\n')
        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

    print('\nSplitting done.')
    print()

    return X_dev, X_test, y_dev, y_test 


def data_partitioner(documents, authors, groups, target, segment=True):
    X = documents
    y = [1 if author.rstrip() == target else 0 for author in authors]

    X_dev, X_test, y_dev, y_test = split(X,y)

    if segment:

        whole_docs_len = len(y_test)

        segmentator_dev = Segmentation(split_policy='by_sentence', tokens_per_fragment=500)
        splitted_docs_dev = segmentator_dev.fit_transform(documents=X_dev, authors=y_dev)
        groups_dev = segmentator_dev.groups

        segmentator_test = Segmentation(split_policy='by_sentence', tokens_per_fragment=500)
        splitted_docs_test = segmentator_test.transform(documents=X_test, authors=y_test)
        groups_test = segmentator_test.groups

        X_dev = splitted_docs_dev[0]
        y_dev = splitted_docs_dev[1]
        groups_dev = segmentator_dev.groups

        X_test = splitted_docs_test[0][:whole_docs_len] # whole_docs_test
        y_test = splitted_docs_test[1][:whole_docs_len] # whole_docs_y_test

        X_test_frag = splitted_docs_test[0][whole_docs_len:]
        y_test_frag = splitted_docs_test[1][whole_docs_len:]
        groups_test_frag = groups_test[whole_docs_len:] 

        print('Segmentation done')

    return X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, groups_test, groups_test_frag


def extract_features(X_dev, X_test, y_dev, X_test_frag):

    latin_function_words = ['et',  'in',  'de',  'ad',  'non',  'ut', 'cum', 'per', 'a', 'sed', 'que', 'quia', 'ex', 'sic',
                        'si', 'etiam', 'idest', 'nam', 'unde', 'ab', 'uel', 'sicut', 'ita', 'enim', 'scilicet', 'nec',
                        'pro', 'autem', 'ibi',  'dum', 'uero', 'tamen', 'inter', 'ideo', 'propter', 'contra', 'sub',
                        'quomodo', 'ubi', 'super', 'iam', 'tam', 'hec', 'post', 'quasi', 'ergo', 'inde', 'e', 'tunc',
                        'atque', 'ac', 'sine', 'nisi', 'nunc', 'quando', 'ne', 'usque', 'siue', 'aut', 'igitur', 'circa',
                        'quidem', 'supra', 'ante', 'adhuc', 'seu' , 'apud', 'olim', 'statim', 'satis', 'ob', 'quoniam',
                        'postea', 'nunquam']


    function_words_extractor = FeaturesFunctionWords(function_words=latin_function_words)
    mendenhall_extractor = FeaturesMendenhall(upto=20)
    words_masker = FeaturesDVEX(function_words=latin_function_words)
    #sentence_len_extractor = FeaturesSentenceLength()
    POS_extractor = FeaturesPOST()
    DEP_extractor = FeaturesDEP()
    punct_extractor = FeaturesPunctuation()


    vectorizers = [
            function_words_extractor,
            words_masker,
            POS_extractor ,
            mendenhall_extractor,
            DEP_extractor,
            #sentence_len_extractor,
            punct_extractor     
        ]
    
    hstacker = HstackFeatureSet(vectorizers)

    feature_sets_dev = []
    feature_sets_test = []
    feature_sets_test_frag = []

    for vectorizer in vectorizers:
        extractor =  FeatureSetReductor(vectorizer)
        feature_sets_dev.append(extractor.fit_transform(X_dev, authors=y_dev))
        feature_sets_test.append(extractor.transform(X_test))
        feature_sets_test_frag.append(extractor.transform(X_test_frag))

    X_dev_stacked = hstacker._hstack(feature_sets_dev)
    X_test_stacked = hstacker._hstack(feature_sets_test)
    X_test_stacked_frag = hstacker._hstack(feature_sets_test_frag)
    
    return X_dev_stacked, X_test_stacked, X_test_stacked_frag

def model_trainer(X_dev_stacked, y_dev, clf=None):
    if not clf:
        clf = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
    #f1_binary = make_scorer(f1_score, pos_label=1)

    clf.fit(X_dev_stacked, y_dev)
    return clf

def get_scores(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=1.0)
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=1.0)
    cf = confusion_matrix(y_test, y_pred).ravel() #(tn, fp, fn, tp)

    if len(y_test) == 1:
        print('Actual:', y_test)
        print('Predicted:', y_pred)
        print()

    print('Precision:', precision)
    print('Recall:', recall)
    print('Accuracy:', acc)
    print('F1:', f1)
    print()
    print(classification_report(y_test, y_pred, zero_division=1.0))
    print()
    print('Confusion matrix: (tn, fp, fn, tp)\n', cf)

    if np.sum(y_test)>1:
        print()
        print('Confusion Matrix:')
        print('TP:', cf[3], '\t|  FP', cf[1])
        print('FN:', cf[2], '\t|  TN', cf[0])

    proba = clf.predict_proba(X_test)

    print('Confidence scores: \n', proba)

    return acc, f1, cf


def save_res(target_author, accuracy, f1, cf, file_name='verifiers_res.csv'):
    #path = '/home/martinaleo/.ssh/authorship/src/Cervantes_base_clf_fullsets-res'
    path= '/home/martinaleo/Quaestio_AV/authorship/src/data/hold_out_res'
    os.chdir(path)
    data = {
        'Target author': target_author,
        'Accuracy':accuracy,
        'F1': f1,
        'TP': cf[3] if len(cf)>1 else 0,
        'FP': cf[1] if len(cf)>1 else 0,
        'FN': cf[2] if len(cf)>1 else 0,
        'TN': cf[0]
    }
    with open(file_name, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        # Check if the file is empty, if so, write the header
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(data)
    print(f"Model res saved in file '{file_name}'")



def build_model(target, save_results=False):

    print('Building model for author', target)

    print('Loading data.\n')
    documents, authors, filenames = load_dataset()
    print('Data loaded.\n')
    #print(np.unique(authors, return_counts=True))

    print('Partitioning data.\n')
    X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, groups_test, groups_test_frag = data_partitioner(documents, authors, filenames, target=target)
    print('Data partitioned.\n')

    print('Extracting features.\n')
    X_dev_stacked, X_test_stacked, X_test_stacked_frag = extract_features(X_dev, X_test, y_dev, X_test_frag)
    print('Features extracted.\n')

    print('Building model.\n')
    clf = model_trainer(X_dev_stacked, y_dev)
    print('Model built.\n')

    print('Evaluating performance.\n')
    acc, f1, cf = get_scores(clf, X_test_stacked, y_test)
    print('Performance evaluated.\n')

    if save_results:
        print('Saving results.\n')
        save_res(target, acc, f1, cf)
        print('Results saved.\n')


    print('Model succsessfully built for author', target)

def loop_over_authors():
    print('Building model for each author.\n')
    _, authors, _ = load_dataset()
    for author in np.unique(authors):
        build_model(target=author, save_results=True)

    

#build_model(target='Michael Scotus')
loop_over_authors()







