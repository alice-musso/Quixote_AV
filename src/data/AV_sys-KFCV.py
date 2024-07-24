import os
import time
from tqdm import tqdm
import csv
import numpy as np
from sklearn.calibration import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from data_loader import load_corpus
from splitting__ import Segmentation
import spacy
from nltk import sent_tokenize
from features import ( 
    DocumentProcessor,
    FeaturesFunctionWords, 
    FeaturesDistortedView, 
    FeaturesMendenhall, 
    FeaturesSentenceLength, 
    FeaturesPOST, 
    FeatureSetReductor,
    FeaturesDEP,
    FeaturesPunctuation,
    HstackFeatureSet,
    FeaturesCharNGram, 
    FeaturesSyllabicQuantities
)
from data_loader import remove_citations

NLP = spacy.load('la_core_web_lg')
TEST_SIZE=0.2
RANDOM_STATE = 42
PROCESSED = True
DEBUG_MODE  = True


def load_dataset(path ='src/data/Quaestio-corpus', remove_test=False, debug_mode=DEBUG_MODE):
    print('Loading data.\n')

    documents, authors, filenames = load_corpus(path=path)

    if debug_mode:
        documents, authors, filenames = documents[:10], authors[:10], filenames[:10]

    print('Data loaded.\n')

    print('Data cleaning.\n')
    documents = [remove_citations(doc) for doc in documents]

    return documents, authors, filenames


# def initialize_language_model(max_length):
#     print('Setting up language model.\n')

#     NLP.max_length = max_length #1364544

#     print('Language model imported.\n')


def split(X, y, target):

    labels = [label for label, _ in y]

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
            neg_X, neg_y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        X_dev = np.concatenate((pos_X, X_dev_neg), axis=0)
        y_dev_ = np.concatenate(([pos_y], y_dev_neg), axis=0)

        X_dev = [str(doc) for doc in X_dev]
        X_test = [str(doc) for doc in X_test]

    else:
        print('\nAuthor with multiple documents\n')
        X_dev, X_test, y_dev_, y_test_ = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
        )
    
    y_dev = [int(label) for label, _ in y_dev_]
    y_test = [int(label) for label, _ in y_test_]

    groups_dev =[title for _, title in y_dev_]
    groups_test =[title for _, title in y_test_]

    pos_docs_idxs = [i for i, label in enumerate(y_dev) if label == 1]
    pos_group_dev = [groups_dev[idx] for idx in pos_docs_idxs]
    pos_docs_idxs2 = [i for i, label in enumerate(y_test) if label == 1]
    pos_group_test = [groups_test[idx] for idx in pos_docs_idxs2]

    print('Target:', target)
    print('Positive training examples:')
    print(', '.join([group.split('-')[1][:-2] for group in pos_group_dev])[1:])
    print('\nPositive test examples:')
    print(', '.join([group.split('-')[1][:-2] for group in pos_group_test])[1:])
    print()
    

    return X_dev, X_test, y_dev, y_test, groups_dev, groups_test 


def data_partitioner(documents, authors, filenames, target, segment=True):
    print('Partitioning data.\n')

    X = documents
    
    y = [(author, filename) for  author, filename in zip(authors, filenames)]
    y = [(1, title) if author.rstrip() == target else (0, title) for author, title in y]
    #y = [1 if author.rstrip() == target else 0 for author in authors]
    

    X_dev, X_test, y_dev, y_test, groups_dev, groups_test = split(X,y, target=target)

    if segment:

        whole_docs_len = len(y_test)

        segmentator_dev = Segmentation(split_policy='by_sentence', tokens_per_fragment=500)
        splitted_docs_dev = segmentator_dev.fit_transform(documents=X_dev, authors=y_dev, filenames=groups_dev)

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


def get_processed_documents(documents, authors, filenames, processed=PROCESSED, cache_file='/home/martinaleo/Quaestio_AV/authorship/.cache/processed_ent_docs_cleaned.pkl'):

    print('Processing documents.\n')

    if not processed:
        NLP.max_length =max([len(document) for document in documents])

        print('Processing docs.\n')
        processor = DocumentProcessor(language_model=NLP, savecache=cache_file)
        processed_docs = processor.process_documents(documents, filenames)
        print('Docs processed. \n')

    else:
        processor = DocumentProcessor(savecache=cache_file)
        processed_docs = processor.process_documents(documents, filenames)

    return processed_docs


def find_segment(segment, processed_document):

    start_segment = sent_tokenize(segment)[0] # language?
    start_idx = processed_document.text.find(start_segment)
    end_idx = start_idx + len(segment)
   
    processed_seg = processed_document.char_span(start_idx, end_idx, alignment_mode='expand')
    if not processed_seg:
        processed_seg = processed_document.char_span(start_idx, end_idx-1, alignment_mode='expand')
        #processed_seg = NLP(segment)
    return processed_seg


def get_processed_segments(processed_docs, X, groups, dataset=''):
    print(f'Extracting processed {dataset} segments.')

    none_count=0
    processed_X = []
    for segment, group in tqdm(zip(X, groups), total=len(X), desc='Progress'):
        if group.endswith('_0_0'): # entire doc
            processed_doc = processed_docs[group[:-4]]
            processed_X.append(processed_doc)

        else: # segment
            group_idx = group.find('_0')
            group_key = group[:group_idx]
            ent_doc_processed = processed_docs[group_key]
            #ent_doc_processed = processed_docs[group[:-2]]
            processed_segment = find_segment(segment, ent_doc_processed)

            if not processed_segment:
                none_count+=1
            processed_X.append(processed_segment)

    print('None count:', none_count)
    print()
    
    return processed_X


def extract_feature_vectors(processed_docs_dev, processed_docs_test, processed_docs_test_frag, y_dev):    #(documents, authors, filenames, nlp, target):

    print('Extracting feature vectors.')

    latin_function_words = ['et',  'in',  'de',  'ad',  'non',  'ut', 'cum', 'per', 'a', 'sed', 'que', 'quia', 'ex', 'sic',
                        'si', 'etiam', 'idest', 'nam', 'unde', 'ab', 'uel', 'sicut', 'ita', 'enim', 'scilicet', 'nec',
                        'pro', 'autem', 'ibi',  'dum', 'uero', 'tamen', 'inter', 'ideo', 'propter', 'contra', 'sub',
                        'quomodo', 'ubi', 'super', 'iam', 'tam', 'hec', 'post', 'quasi', 'ergo', 'inde', 'e', 'tunc',
                        'atque', 'ac', 'sine', 'nisi', 'nunc', 'quando', 'ne', 'usque', 'siue', 'aut', 'igitur', 'circa',
                        'quidem', 'supra', 'ante', 'adhuc', 'seu' , 'apud', 'olim', 'statim', 'satis', 'ob', 'quoniam',
                        'postea', 'nunquam']


    function_words_vectorizer = FeaturesFunctionWords(function_words=latin_function_words)
    mendenhall_vectorizer = FeaturesMendenhall(upto=20)
    words_masker_SA = FeaturesDistortedView(function_words=latin_function_words, method='DVSA')
    words_masker_MA = FeaturesDistortedView(function_words=latin_function_words, method='DVMA')
    words_masker_EX = FeaturesDistortedView(function_words=latin_function_words, method='DVEX')
    sentence_len_extractor = FeaturesSentenceLength()
    POS_vectorizer = FeaturesPOST()
    DEP_vectorizer = FeaturesDEP()
    punct_vectorizer = FeaturesPunctuation()
    char_extractor = FeaturesCharNGram()
    syllabic_quant_extractor = FeaturesSyllabicQuantities()


    vectorizers = [
            # words_masker_SA,
            words_masker_MA,
            # words_masker_EX,
            syllabic_quant_extractor,
            function_words_vectorizer,
            POS_vectorizer ,
            mendenhall_vectorizer,
            DEP_vectorizer,
            sentence_len_extractor,
            punct_vectorizer,
            char_extractor     
        ]
    
    hstacker = HstackFeatureSet(vectorizers)

    feature_sets_dev = []
    feature_sets_test = []
    feature_sets_test_frag = []

    for vectorizer in vectorizers:
        #extractor =  FeatureSetReductor(vectorizer)
        print('\nExtracting',vectorizer)

        reductor =  FeatureSetReductor(vectorizer, k_ratio=0.5)

        print('\nProcessing development set')
        features_dev = reductor.fit_transform(processed_docs_dev, y_dev)
        feature_sets_dev.append(features_dev)

        print('\nProcessing test set')
        features_test = reductor.transform(processed_docs_test)
        feature_sets_test.append(features_test)

        print('\nProcessing test set segments')
        features_test_frag = reductor.transform(processed_docs_test_frag)
        feature_sets_test_frag.append(features_test_frag)
        
    X_dev_stacked = hstacker._hstack(feature_sets_dev)
    X_test_stacked = hstacker._hstack(feature_sets_test)
    X_test_stacked_frag = hstacker._hstack(feature_sets_test_frag)

    print('\nFeature vectors extracted.\n')
    
    return X_dev_stacked, X_test_stacked, X_test_stacked_frag


def prepare_data(target):
    documents, authors, filenames = load_dataset()
    filenames = [filename+'_0' for filename in filenames]

    processed_documents = get_processed_documents(documents, authors, filenames)

    X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, groups_test, groups_test_frag = data_partitioner(documents, authors, filenames, target=target)

    X_dev_processed = get_processed_segments(processed_documents, X_dev, groups_dev, dataset='training')
    X_test_processed = get_processed_segments(processed_documents, X_test, groups_test, dataset='test')
    X_test_frag_processed = get_processed_segments(processed_documents, X_test_frag, groups_test_frag, dataset='test fragments')

    X_dev_stacked, X_test_stacked, X_test_stacked_frag = extract_feature_vectors(X_dev_processed, X_test_processed, X_test_frag_processed, y_dev)

    return X_dev_stacked, X_test_stacked, y_dev, y_test, X_test_stacked_frag, y_test_frag, groups_dev, groups_test, groups_test_frag


def model_trainer(X_dev_stacked, y_dev, groups_dev, model, model_name):

    groups_dev = [filename[:filename.find('_0')] for filename in groups_dev]

    if model_name in ['Linear SVC', 'Logistic Regressor', 'Probabilistic SVC']:
        param_grid = {'C': np.logspace(-4,4,9)}
    elif model_name == 'Adaboost':
        param_grid = {'learning_rate': [1.0],
                      'n_estimators' : [500]}

    grid = GridSearchCV(model,
                        param_grid=param_grid,
                        cv=5,
                        n_jobs=-1,
                        scoring='f1',
                        verbose=True)    

    grid.fit(X_dev_stacked, y_dev, groups=groups_dev)

    print('Model fitted. Best params:')
    print(grid.best_params_)
    print()

    print('Model built. \n')
    return grid.best_estimator_


def get_scores(clf, X_test, y_test, groups_test):
    print('Evaluating performance...', '(on fragmented text)' if len(y_test) > 110 else '\n')

    y_pred = clf.predict(X_test)
    y_pred=[int(pred) for pred in y_pred]
    # print('Actual:', [(label, filename) for label,filename in zip(y_test, groups_test)])
    # print('Predicted:', y_pred)
    # print()

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
    print('Confusion matrix: (tn, fp, fn, tp)\n', cf, '\n')

    # if np.sum(y_test)>1:
    #     print()
    #     print('Confusion Matrix:')
    #     print('TP:', cf[3], '\t|  FP', cf[1])
    #     print('FN:', cf[2], '\t|  TN', cf[0])

    #proba = clf.predict_proba(X_test)

    #print('Confidence scores: \n', proba)

    return acc, f1, cf


def save_res(target_author, accuracy, f1, cf, model_name, file_name='verifiers_res_kfcv.csv'):
    path= '/home/martinaleo/Quaestio_AV/authorship/src/data/hold_out_res'
    print(f'Saving results in {file_name}\n')

    os.chdir(path)
    data = {
        'Target author': target_author,
        'Model': model_name,
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
    print(f"{model_name} res for author {target_author} saved in file '{file_name}'\n")


def build_model(target, save_results=True):
    hour = '0' + str(time.localtime()[3]) if len(str(time.localtime()[3])) == 1 else str(time.localtime()[3])
    minutes = '0' + str(time.localtime()[4]) if len(str(time.localtime()[4])) == 1 else str(time.localtime()[4])

    print('Start time:', hour + ':'+ minutes, '\n')
    print('Building model for author', target + '.\n')

    start_time = time.time()

    X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, groups_test, groups_test_frag = prepare_data(target=target)
    
    models = [
        #(LinearSVC(random_state=RANDOM_STATE, dual='auto'), 'Linear SVC'),
        #(LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1), 'Logistic Regressor'),
        #(SVC(kernel='linear', probability=True, random_state=RANDOM_STATE), 'Probabilistic SVC')
        (AdaBoostClassifier(random_state=RANDOM_STATE), 'Adaboost')
    ]

    for model, model_name in models:
        print(f'Building {model_name} classifier...\n')
        clf = model_trainer(X_dev, y_dev, groups_dev, model=model, model_name=model_name)
        acc, f1, cf = get_scores(clf, X_test, y_test, groups_test)
        acc_frag, f1_frag, cf_frag = get_scores(clf, X_test_frag, y_test_frag, groups_test_frag)

        if save_results:
            save_res(target, acc, f1, cf, model_name)

    print(f'Time spent for model building for author {target}:', round((time.time() - start_time)/60, 2), 'minutes.')


def loop_over_authors():
    _, authors, _ = load_corpus(path='/home/martinaleo/Quaestio_AV/authorship/src/data/Quaestio-corpus')
    for author in np.unique(authors):
        if author not in ['Anonymus', 'Misc']:
            build_model(target=author)

#loop_over_authors()
            
build_model(target='Dante')

# Fitting 5 folds for each of 28 candidates, totalling 140 fits
# Model fitted. Best params:
# {'learning_rate': 1.0, 'n_estimators': 500}, sel k con k_ratio=0.5 e segmenti da 500 token
# 1 tp 2 fn
# riconosce epistola 12 ma non il devulgari

