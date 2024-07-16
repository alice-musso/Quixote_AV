from tqdm import tqdm
import time

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

from splitting__ import Segmentation

from helpers__ import get_function_words

from features import ( 
    DocumentProcessor,
    FeaturesFunctionWords, 
    FeaturesDVEX, 
    FeaturesMendenhall, 
    FeaturesSentenceLength, 
    FeaturesPOST, 
    FeatureSetReductor,
    FeaturesDEP,
    FeaturesPunctuation,
    HstackFeatureSet,
    FeaturesCharNGram
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
    print('Loading data.\n')

    path = '/home/martinaleo/Quaestio_AV/authorship/src/data/Quaestio-corpus'
    documents, authors, filenames = load_corpus(path=path)

    print('Data loaded.\n')

    '''code to remove pattern missing'''
    return documents, authors, filenames


def split(X, y):

    labels = [label for label, title in y]

    if np.sum(labels) < 2:
        print()
        print('One doc only author')
        print()

        positive_doc_idx = labels.index(1)
        pos_X = X[positive_doc_idx]
        pos_X = np.expand_dims(pos_X, axis=0)
        pos_y = y[positive_doc_idx]

        neg_X = np.delete(X, positive_doc_idx)
        neg_y = y[:positive_doc_idx] + y[positive_doc_idx+1:] #np.delete(y, positive_doc_idx)
        
        X_dev_neg, X_test, y_dev_neg, y_test_ = train_test_split(
            neg_X, neg_y, test_size=0.3, random_state=42
        )
        X_dev = np.concatenate((pos_X, X_dev_neg), axis=0)
        y_dev_ = np.concatenate(([pos_y], y_dev_neg), axis=0)

        X_dev = [str(doc) for doc in X_dev]
        #y_dev = list(y_dev)
        #y_dev = [label for label, _ in y_dev_]

        X_test = [str(doc) for doc in X_test]
        #y_test = list(y_test)
        #y_test = [label for label, _ in y_test_]

        # groups_dev =[title for _, title in y_dev_]
        # groups_test =[title for _, title in y_test_]

    else:
        print('\nAuthor with multiple documents\n')
        X_dev, X_test, y_dev_, y_test_ = train_test_split(
            X, y, test_size=0.3, stratify=labels, random_state=42
        )
    
    y_dev = [label for label, _ in y_dev_]
    y_test = [label for label, _ in y_test_]

    groups_dev =[title for _, title in y_dev_]
    groups_test =[title for _, title in y_test_]

    return X_dev, X_test, y_dev, y_test, groups_dev, groups_test 


def data_partitioner(documents, authors, filenames, target, segment=True):
    print('Partitioning data.\n')

    X = documents
    # label encoder per salvare info relative a titoli
    y = [(author, filename.split('-')[1].strip()) for  author, filename in zip(authors, filenames)]
    y = [(1, title) if author.rstrip() == target else (0, title) for author, title in y]
    #y = [1 if author.rstrip() == target else 0 for author in authors]
    

    X_dev, X_test, y_dev, y_test, groups_dev, groups_test = split(X,y)

    if segment:

        whole_docs_len = len(y_test)

        segmentator_dev = Segmentation(split_policy='by_sentence', tokens_per_fragment=500)
        splitted_docs_dev = segmentator_dev.fit_transform(documents=X_dev, authors=y_dev, filenames=groups_dev)
        #groups_dev = segmentator_dev.groups

        segmentator_test = Segmentation(split_policy='by_sentence', tokens_per_fragment=500)
        splitted_docs_test = segmentator_test.transform(documents=X_test, authors=y_test, filenames=groups_test)
        groups_test = segmentator_test.groups

        X_dev = splitted_docs_dev[0]
        y_dev = splitted_docs_dev[1]
        groups_dev = segmentator_dev.groups

        X_test = splitted_docs_test[0][:whole_docs_len] # whole_docs_test
        y_test = splitted_docs_test[1][:whole_docs_len] # whole_docs_y_test
        groups_test_entire_docs = groups_test[:whole_docs_len]

        X_test_frag = splitted_docs_test[0][whole_docs_len:]
        y_test_frag = splitted_docs_test[1][whole_docs_len:]
        groups_test_frag = groups_test[whole_docs_len:] 

        print('Segmentation done.\n')

    print('Data partitioned.\n')

    return X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, groups_test_entire_docs, groups_test_frag


def initialize_language_model(max_length):
    print('Importing language model.\n')

    nlp = spacy.load('la_core_web_lg')
    nlp.max_length = max_length #1364544

    print('Language model imported.\n')
    return nlp


# def extract_linguistic_features(nlp, documents, partition='Training set'):
#     print(f'Extracting linguistic features for {partition}.\n')

#     sentences_tot = []
#     tokens_tot = []
#     POS_tags_tot = []
#     DEP_tags_tot = []

#     for doc in documents:
#         processed_doc = nlp(doc)

#         sentences = [str(sentence) for sentence in processed_doc.sents]
#         tokens = [str(token) for token in processed_doc]
#         POS_tags = [token.pos_ if token.pos_ != '' else 'Unk' for token in processed_doc]
#         DEP_tags = [token.dep_ if token.dep_ != '' else 'Unk' for token in processed_doc]

#         sentences_tot += sentences
#         tokens_tot += tokens
#         POS_tags_tot += POS_tags
#         DEP_tags_tot += DEP_tags

#     print('linguistic features extracted.\n')

#     return sentences_tot, tokens_tot, POS_tags_tot, DEP_tags_tot
    



def extract_feature_vectors(documents, authors, filenames, nlp, target):

    X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, groups_test, groups_test_frag = data_partitioner(documents, authors, filenames, target=target)

    print('Extracting linguistic features. \n')
    
    processor = DocumentProcessor(language_model=nlp)
    processed_docs_dev = processor.process_documents(X_dev, groups_dev)
    processed_docs_test = processor.process_documents(X_test, groups_test)
    processed_docs_test_frag = processor.process_documents(X_test_frag, groups_test_frag)

    print('Linguistic features extracted. \n')

    print('Extracting feature vectors.\n')

    latin_function_words = ['et',  'in',  'de',  'ad',  'non',  'ut', 'cum', 'per', 'a', 'sed', 'que', 'quia', 'ex', 'sic',
                        'si', 'etiam', 'idest', 'nam', 'unde', 'ab', 'uel', 'sicut', 'ita', 'enim', 'scilicet', 'nec',
                        'pro', 'autem', 'ibi',  'dum', 'uero', 'tamen', 'inter', 'ideo', 'propter', 'contra', 'sub',
                        'quomodo', 'ubi', 'super', 'iam', 'tam', 'hec', 'post', 'quasi', 'ergo', 'inde', 'e', 'tunc',
                        'atque', 'ac', 'sine', 'nisi', 'nunc', 'quando', 'ne', 'usque', 'siue', 'aut', 'igitur', 'circa',
                        'quidem', 'supra', 'ante', 'adhuc', 'seu' , 'apud', 'olim', 'statim', 'satis', 'ob', 'quoniam',
                        'postea', 'nunquam']


    function_words_vectorizer = FeaturesFunctionWords(function_words=latin_function_words)
    mendenhall_vectorizer = FeaturesMendenhall(upto=20)
    words_masker = FeaturesDVEX(function_words=latin_function_words)
    #sentence_len_extractor = FeaturesSentenceLength()
    POS_vectorizer = FeaturesPOST()
    DEP_vectorizer = FeaturesDEP()
    punct_vectorizer = FeaturesPunctuation()


    vectorizers = [
            function_words_vectorizer,
            words_masker,
            POS_vectorizer ,
            mendenhall_vectorizer,
            DEP_vectorizer,
            #sentence_len_extractor,
            punct_vectorizer     
        ]
    
    hstacker = HstackFeatureSet(vectorizers)

    feature_sets_dev = []
    feature_sets_test = []
    feature_sets_test_frag = []

    for vectorizer in vectorizers:
        #extractor =  FeatureSetReductor(vectorizer)
        print(vectorizer)

        features_dev = vectorizer.fit_transform(processed_docs_dev)
        feature_sets_dev.append(features_dev)

        features_test = vectorizer.transform(processed_docs_test)
        feature_sets_test.append(features_test)

        features_test_frag = vectorizer.transform(processed_docs_test_frag)
        feature_sets_test_frag.append(features_test_frag)
        
        # feature_sets_dev.append(extractor.fit_transform(processed_docs_dev))
        # feature_sets_test.append(extractor.transform(processed_docs_test))
        # feature_sets_test_frag.append(extractor.transform(processed_docs_test_frag))

    X_dev_stacked = hstacker._hstack(feature_sets_dev)
    X_test_stacked = hstacker._hstack(feature_sets_test)
    X_test_stacked_frag = hstacker._hstack(feature_sets_test_frag)

    print('Feature vectors extracted')
    
    return X_dev_stacked, X_test_stacked, X_test_stacked_frag, y_dev, y_test


def model_trainer(X_dev_stacked, y_dev, clf=None):
    print('Building model...\n')
    if not clf:
        clf = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')

    clf.fit(X_dev_stacked, y_dev)

    print('Model built. \n')
    return clf

def get_scores(clf, X_test, y_test):
    print('Evaluating performance...\n')

    y_pred = clf.predict(X_test)
    y_pred=[int(pred) for pred in y_pred]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=1.0)# pos_label='1')
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

    # if np.sum(y_test)>1:
    #     print()
    #     print('Confusion Matrix:')
    #     print('TP:', cf[3], '\t|  FP', cf[1])
    #     print('FN:', cf[2], '\t|  TN', cf[0])

    proba = clf.predict_proba(X_test)

    print('Confidence scores: \n', proba)

    print('Performance evaluated.\n')
    return acc, f1, cf


def save_res(target_author, accuracy, f1, cf, file_name='verifiers_res_tot.csv'):
    path= '/home/martinaleo/Quaestio_AV/authorship/src/data/hold_out_res'
    print(f'Saving results in {file_name}\n')

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
    print('Start time:', time.localtime(), '\n')
    print('Building model for author', target + '.\n')

    start_time = time.time()

    documents, authors, filenames = load_dataset()

    #documents, authors, filenames = documents[:10], authors[:10], filenames[:10]

    nlp = initialize_language_model(max_length=max([len(document) for document in documents]))

    X_dev_stacked, X_test_stacked, X_test_stacked_frag, y_dev, y_test = extract_feature_vectors(documents, authors, filenames, nlp, target)

    #print(X_dev_stacked.shape, X_test_stacked.shape, len(y_dev), len(y_test))

    clf = model_trainer(X_dev_stacked, y_dev)

    acc, f1, cf = get_scores(clf, X_test_stacked, y_test)

    if save_results:
        save_res(target, acc, f1, cf)
        
    print('Model succsessfully built for author', target)
    print(f'Time spent for model building for author {target}:', time.time() - start_time)

def loop_over_authors():
    print('Building model for each author.\n')
    _, authors, _ = load_dataset()
    for author in np.unique(authors):
        build_model(target=author, save_results=True)

    
#build_model(target='Antonio Pelacani da Parma')
loop_over_authors()