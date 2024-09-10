import os
from pathlib import Path
import pickle
import time
from tqdm import tqdm
import csv
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from dro import DistributionalRandomOversampling
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from sklearn.calibration import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
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
from mapie.classification import MapieClassifier

NLP = spacy.load('la_core_web_lg')
#TEST_SIZE = 0.3
SEGMENT_MIN_TOKEN_SIZE = 400
RANDOM_STATE = 42
PROCESSED = False # whether the linguistic features have been already extracted or not
LOAD_DATA = False # whether the vector space representation have been already obtained or not
K_RATIO = 1.0
OVERSAMPLE = False
TARGET='Dante'
PARTITIONED_DATA_CACHE_FILE = f'.partitioned_data_cache/partitioned_data_{TARGET}.pkl'
DEBUG_MODE  = False


def load_dataset(path ='src/data/Quaestio-corpus', debug_mode=DEBUG_MODE):
    print('Loading data.\n')

    documents, authors, filenames = load_corpus(path=path, remove_epistles=False, remove_test=True, remove_egloghe=False)

    if debug_mode:
        documents, authors, filenames = documents[:10], authors[:10], filenames[:10]

    print('Data loaded.\n')

    # if remove_test:
    #     documents, authors, filenames = documents[:-1], authors[:-1], filenames[:-1]
    print([filename for filename in filenames if 'dante' in filename.lower()])

    # print('Data cleaning.\n')
    # documents = [remove_citations(doc) for doc in documents]

    return documents, authors, filenames


# def initialize_language_model(max_length):
#     print('Setting up language model.\n')

#     NLP.max_length = max_length #1364544

#     print('Language model imported.\n')


# def split(X, y, target):

#     labels = [label for label, _ in y]

#     if np.sum(labels) < 2:
#         print()
#         print('One doc only author')
#         print()

#         positive_doc_idx = labels.index(1)
#         pos_X = X[positive_doc_idx]
#         pos_X = np.expand_dims(pos_X, axis=0)
#         pos_y = y[positive_doc_idx]

#         neg_X = np.delete(X, positive_doc_idx)
#         neg_y = y[:positive_doc_idx] + y[positive_doc_idx+1:] #np.delete(y, positive_doc_idx)
        
#         X_dev_neg, X_test, y_dev_neg, y_test_ = train_test_split(
#             neg_X, neg_y, test_size=TEST_SIZE, random_state=RANDOM_STATE
#         )
#         X_dev = np.concatenate((pos_X, X_dev_neg), axis=0)
#         y_dev_ = np.concatenate(([pos_y], y_dev_neg), axis=0)

#         X_dev = [str(doc) for doc in X_dev]
#         X_test = [str(doc) for doc in X_test]

#     else:
#         print('\nAuthor with multiple documents\n')
#         X_dev, X_test, y_dev_, y_test_ = train_test_split(
#             X, y, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
#         )
    
#     y_dev = [int(label) for label, _ in y_dev_]
#     y_test = [int(label) for label, _ in y_test_]

#     groups_dev =[title for _, title in y_dev_]
#     groups_test =[title for _, title in y_test_]

#     pos_docs_idxs = [i for i, label in enumerate(y_dev) if label == 1]
#     pos_group_dev = [groups_dev[idx] for idx in pos_docs_idxs]
#     pos_docs_idxs2 = [i for i, label in enumerate(y_test) if label == 1]
#     pos_group_test = [groups_test[idx] for idx in pos_docs_idxs2]

#     print('Target:', target)
#     print('Positive training examples:')
#     print(', '.join([group.split('-')[1][:-2] for group in pos_group_dev])[1:])
#     print('\nPositive test examples:')
#     print(', '.join([group.split('-')[1][:-2] for group in pos_group_test])[1:])
#     print()
    

#     return X_dev, X_test, y_dev, y_test, groups_dev, groups_test 


def oversample_positive_class(X_dev, X_test, y_dev, y_test, method='SMOTE'):
    print(f'Oversampling minority class using {method} method')

    if method =='DRO':
        
        y_dev = np.array(y_dev)
        y_test = np.array(y_test)

        train_nwords = np.asarray(X_dev.sum(axis=1).getA().flatten(), dtype=int)
        test_nwords = np.asarray(X_test.sum(axis=1).getA().flatten(), dtype=int)

        positives = y_dev.sum()
        nD = len(y_dev)
        print('Before oversampling')
        print(f'positives = {positives} (prevalence={positives*100/nD:.2f}%)')

        dro = DistributionalRandomOversampling(rebalance_ratio=0.2)
        X_dev, y_dev = dro.fit_transform(X_dev, y_dev, train_nwords)

        
        #samples_test = dro._samples_to_match_ratio(y_test)
        X_test = dro.transform(X_test, test_nwords, samples=1)
        #y_test = dro._oversampling_observed(y_test, samples_test)

        positives = y_dev.sum()
        nD = len(y_dev)
        print('After oversampling')
        print(f'positives = {positives} (prevalence={positives*100/nD:.2f}%)')
        print(X_dev.shape, len(y_dev))
        print(X_test.shape, len(y_test))
    
    elif method == 'SMOTE':
        print('Original dataset shape %s' % Counter(y_dev))
        sm = BorderlineSMOTE( sampling_strategy=0.2,random_state=RANDOM_STATE)
        X_dev, y_dev = sm.fit_resample(X_dev, y_dev)
        print('Resampled dataset shape %s' % Counter(y_dev))

    elif method == 'ADASYN':
        print('Original dataset shape %s' % Counter(y_dev))
        sm = ADASYN(sampling_strategy=0.2, random_state=RANDOM_STATE)
        X_dev, y_dev = sm.fit_resample(X_dev, y_dev)
        print('Resampled dataset shape %s' % Counter(y_dev))

    return X_dev, X_test, y_dev, y_test


def LOO_split(i, X, y, doc, ylabel, filenames):

    doc_name = filenames[i]
    print('Test document:', doc_name)
    X_test = [doc]
    y_test = [ylabel]
    X_dev = list(np.delete(X, i))
    y_dev = list(np.delete(y, i))
    groups_dev = list(np.delete(filenames, i))
    return X_dev, X_test, y_dev, y_test, groups_dev, [doc_name] # doc_name==groups_test


#def data_partitioner(documents, authors, filenames, target, segment=True, oversample=True):
def segment_data(X_dev, X_test, y_dev, y_test, groups_dev, groups_test):
    print('Data Segmentation.\n')

    # X = documents
    # y = [1 if author.rstrip() == target else 0 for author in authors]
    # #y = [1 if author.rstrip() == target else 0 for author in authors]
    
    # for i, (doc, ylabel) in enumerate(zip(X,y)):
    #     X_dev, X_test, y_dev, y_test, groups_dev, groups_test = LOO_split(i, X, y, doc, ylabel, filenames)

    whole_docs_len = len(y_test)

    segmentator_dev = Segmentation(split_policy='by_sentence', tokens_per_fragment=SEGMENT_MIN_TOKEN_SIZE)
    splitted_docs_dev = segmentator_dev.fit_transform(documents=X_dev, authors=y_dev, filenames=groups_dev)

    segmentator_test = Segmentation(split_policy='by_sentence', tokens_per_fragment=SEGMENT_MIN_TOKEN_SIZE)
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


def get_processed_documents(documents, authors, filenames, processed=PROCESSED, cache_file='/home/martinaleo/Quaestio_AV/authorship/.cache/processed_docs_def.pkl'): #processed_ent_docs_cleaned

    print('Processing documents.\n')

    if not processed:
        NLP.max_length = max([len(document) for document in documents])

        print('Processing docs.\n')
        processor = DocumentProcessor(language_model=NLP, savecache=cache_file)
        #processor.delete_doc('Dante - Quaestio_0')
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
    char_extractor = FeaturesCharNGram(n=(2,3))
    syllabic_quant_extractor = FeaturesSyllabicQuantities()


    vectorizers = [
            # words_masker_SA,
            words_masker_MA,
            # words_masker_EX,
            # syllabic_quant_extractor,
            # function_words_vectorizer,
            # POS_vectorizer ,
            mendenhall_vectorizer,
            # DEP_vectorizer,
            # sentence_len_extractor,
            # punct_vectorizer,
            char_extractor     
        ]
    
    hstacker = HstackFeatureSet(vectorizers)

    feature_sets_dev = []
    feature_sets_test = []
    feature_sets_test_frag = []

    for vectorizer in vectorizers:
        #extractor =  FeatureSetReductor(vectorizer)
        print('\nExtracting',vectorizer)

        reductor =  FeatureSetReductor(vectorizer, k_ratio=K_RATIO)

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


# def prepare_data(target, oversample=OVERSAMPLE, store_data=True):

#     # documents, authors, filenames = load_dataset()
#     # filenames = [filename+'_0' for filename in filenames]

#     # processed_documents = get_processed_documents(documents, authors, filenames)

#     X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, groups_test, groups_test_frag = data_partitioner(documents, authors, filenames, target=target)

#     X_dev_processed = get_processed_segments(processed_documents, X_dev, groups_dev, dataset='training')
#     X_test_processed = get_processed_segments(processed_documents, X_test, groups_test, dataset='test')
#     X_test_frag_processed = get_processed_segments(processed_documents, X_test_frag, groups_test_frag, dataset='test fragments')

#     X_dev, X_test, X_test_frag = extract_feature_vectors(X_dev_processed, X_test_processed, X_test_frag_processed, y_dev)
    
#     if oversample:
#         X_dev, X_test, y_dev, y_test = oversample_positive_class(X_dev, X_test, y_dev, y_test)

#     if store_data:
#         data = {
#             "X_dev" : X_dev, 
#             "X_test" : X_test, 
#             "y_dev" : y_dev, 
#             "y_test" : y_test,  
#             "X_test_frag" : X_test_frag, 
#             "y_test_frag" : y_test_frag, 
#             "groups_dev" : groups_dev, 
#             "groups_test" : groups_test, 
#             "groups_test_frag" : groups_test_frag
#         }

#         store_partitioned_data(data)

#     return X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, groups_test, groups_test_frag


def store_partitioned_data(data, cache_file=PARTITIONED_DATA_CACHE_FILE):
    if os.path.exists(cache_file):
            print('\nData already stored.')
    else:
        print(f'\nStoring partitioned data in {cache_file}\n')
        cache = {}

        for dataset_name, dataset in data.items():
            cache[dataset_name] = dataset

        parent = Path(cache_file).parent
        if parent:
            os.makedirs(parent, exist_ok=True)
        pickle.dump(cache, open(cache_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Cache successfully stored in {cache_file}\n')


def load_partitioned_data(cache_file=PARTITIONED_DATA_CACHE_FILE):
    print(f'\nLoading cache from {cache_file}')
    data = pickle.load(open(cache_file, 'rb'))
    datasets = []
    for _, dataset in data.items():
        datasets.append(dataset)
    print('Data loaded.\n')
    return datasets


def model_trainer(X_dev_stacked, y_dev, groups_dev, model, model_name):

    groups_dev = [filename[:filename.find('_0')] for filename in groups_dev]

    if model_name in ['Linear SVC', 'Logistic Regressor', 'Probabilistic SVC']:
        param_grid = {'C': np.logspace(-4,4,9)}
    elif model_name == 'Adaboost':
        param_grid = {'learning_rate': [1.0],
                      'n_estimators' : [50],
                      'estimator__C': [1]}#np.logspace(-4,4,9)}
    elif model_name == 'Linear SVC with calibration':
        param_grid = {'estimator__C': np.logspace(-4,4,9)}


    grid = GridSearchCV(model,
                        param_grid=param_grid,
                        cv=5,
                        n_jobs=-1,
                        scoring='accuracy',
                        verbose=True)    

    grid.fit(X_dev_stacked, y_dev)#, groups=groups_dev)

    print('Model fitted. Best params:')
    print(grid.best_params_)
    print()

    print('Model built. \n')
    return grid.best_estimator_


def get_scores(clf, X_test, y_test, groups_test, y_dev,return_proba=True):
    print('Evaluating performance...', '(on fragmented text)\n' if len(y_test) > 110 else '\n')

    y_pred = clf.predict(X_test)
    print('Actual:', y_test)
    print('Predicted:', y_pred)
    print()

    if return_proba:
        probabilities = clf.predict_proba(X_test)
        posterior_proba = max(probabilities[0])
        print('Posterior probability:', max(probabilities[0]))

        class_labels = clf.classes_
        sorted_probs = sorted(zip(class_labels, probabilities[0]), key=lambda x: x[1], reverse=True)
            
        print("  Probabilities (sorted):")
        for label, prob in sorted_probs:
            print(f"{label}: {prob:.4f}")
        print()
        
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1.0)# pos_label='1')
    precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=1.0)
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

    return acc, f1, cf, posterior_proba


def save_res(target_author, accuracy, f1, posterior_proba, cf, model_name, doc_name, file_name='verifiers_res_LOO_best_config3.csv'):
    path= '/home/martinaleo/Quaestio_AV/authorship/src/data/hold_out_res'
    print(f'Saving results in {file_name}\n')

    os.chdir(path)
    data = {
        'Target author': target_author,
        'Document test': doc_name,
        'Model': model_name,
        'Accuracy':accuracy,
        'F1': f1,
        'Proba': posterior_proba,
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


def build_model(target, save_results=True, oversample=OVERSAMPLE):

    hour = '0' + str(time.localtime()[3]) if len(str(time.localtime()[3])) == 1 else str(time.localtime()[3])
    minutes = '0' + str(time.localtime()[4]) if len(str(time.localtime()[4])) == 1 else str(time.localtime()[4])

    print('Start time:', hour + ':'+ minutes, '\n')
    print('Building LOO model for author', target + '.\n')

    start_time = time.time()

    documents, authors, filenames = load_dataset()
    filenames = [filename+'_0' for filename in filenames]


    processed_documents = get_processed_documents(documents, authors, filenames)
    y = [author for author in authors]

    idx_dante=[31,3,40,17,41,18,32,38,63,39,36,37,62,64]

    for i, (doc, ylabel) in enumerate(zip(documents,y)):
        #if i in idx_dante:
        if 'monarchia' in filenames[i].lower(): # target text ' quaestio'
            start_time_single_iteration = time.time()

            # LOO_split(i, X, y, doc, ylabel, filenames)
            X_dev, X_test, y_dev, y_test, groups_dev, groups_test = LOO_split(i, documents, y, doc, ylabel, filenames)

            # segment_data(X_dev, X_test, y_dev, y_test, groups_dev, groups_test)

            X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, groups_test, groups_test_frag = segment_data(X_dev, X_test, y_dev, y_test, groups_dev, groups_test)

            X_dev_processed = get_processed_segments(processed_documents, X_dev, groups_dev, dataset='training')
            X_test_processed = get_processed_segments(processed_documents, X_test, groups_test, dataset='test')
            X_test_frag_processed = get_processed_segments(processed_documents, X_test_frag, groups_test_frag, dataset='test fragments')

            X_dev, X_test, X_test_frag = extract_feature_vectors(X_dev_processed, X_test_processed, X_test_frag_processed, y_dev)
            
            if oversample:
                X_dev, X_test, y_dev, y_test = oversample_positive_class(X_dev, X_test, y_dev, y_test)
        
        
            models = [
                # (LinearSVC(random_state=RANDOM_STATE, dual='auto'), 'Linear SVC'),
                (LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1), 'Logistic Regressor'),
                # (SVC(kernel='linear', probability=True, random_state=RANDOM_STATE), 'Probabilistic SVC'),
                # (CalibratedClassifierCV(estimator=LinearSVC(random_state=RANDOM_STATE, dual='auto'), cv=5, n_jobs=-1), 'Linear SVC with calibration'),
                #(AdaBoostClassifier(estimator=LinearSVC(random_state=RANDOM_STATE, dual='auto'), algorithm='SAMME', random_state=RANDOM_STATE), 'Adaboost')
            ]

            for model, model_name in models:
                print(f'\nBuilding {model_name} classifier...\n')
                clf = model_trainer(X_dev, y_dev, groups_dev, model=model, model_name=model_name)
                acc, f1, cf, posterior_proba = get_scores(clf, X_test, y_test, groups_test, y_dev)
                # acc_frag, f1_frag, cf_frag = get_scores(clf, X_test_frag, y_test_frag, groups_test_frag)


                if save_results:
                    save_res(target, acc, f1, posterior_proba, cf, model_name, groups_test[0][:-2])

            print(f'Time spent for model building for document {(groups_test[0][:-2])}:', round((time.time() - start_time_single_iteration)/60, 2), 'minutes.')

    print(f'Time spent for model building for author {target}:', round((time.time() - start_time)/60, 2), 'minutes.')


def loop_over_authors():
    _, authors, _ = load_corpus(path='/home/martinaleo/Quaestio_AV/authorship/src/data/Quaestio-corpus')
    for author in np.unique(authors):
        if author not in ['Anonymus', 'Misc']:
            build_model(target=author)

#loop_over_authors()
            
build_model(target=TARGET)

# Fitting 5 folds for each of 28 candidates, totalling 140 fits
# Model fitted. Best params:
# {'learning_rate': 1.0, 'n_estimators': 500}, sel k con k_ratio=0.5 e segmenti da 500 token
# 1 tp 2 fn
# riconosce epistola 12 ma non il devulgari