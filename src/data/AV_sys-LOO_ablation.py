import os
from pathlib import Path
import pickle
import time
from tqdm import tqdm
import csv
import numpy as np
from collections import Counter
from scipy.sparse import hstack, vstack, issparse
from dro import DistributionalRandomOversampling
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from sklearn.calibration import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold, train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from data_loader import load_corpus
from splitting__ import Segmentation
import spacy
import nltk
from nltk import sent_tokenize
from features import ( 
    DocumentProcessor,
    # ForwardFeatureSelector,
    # BackwardFeatureSelector,
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
from embeddings_extraction import load_embeddings
from ablation_strategies import ForwardFeatureSelector, BackwardFeatureSelector

NLP = spacy.load('la_core_web_lg')
N_JOBS = 32
SEGMENT_MIN_TOKEN_SIZE = 400
RANDOM_STATE = 42
PROCESSED = False # whether the linguistic features have been already extracted or not
K_RATIO = 1.0
OVERSAMPLE = True
TARGET='Dante'
PARTITIONED_DATA_CACHE_FILE = f'.partitioned_data_cache/partitioned_data_{TARGET}.pkl'
DEBUG_MODE  = False

TEST_DOCUMENT = 'Dante - epistola4' + '_0'

REMOVE_TEST=False if TEST_DOCUMENT == 'Dante - Quaestio_0' else True
REMOVE_EGLOGHE = True
EXTRACT_EMBEDDINGS=True

latin_function_words = ['et',  'in',  'de',  'ad',  'non',  'ut', 'cum', 'per', 'a', 'sed', 'que', 'quia', 'ex', 'sic',
                        'si', 'etiam', 'idest', 'nam', 'unde', 'ab', 'uel', 'sicut', 'ita', 'enim', 'scilicet', 'nec',
                        'pro', 'autem', 'ibi',  'dum', 'uero', 'tamen', 'inter', 'ideo', 'propter', 'contra', 'sub',
                        'quomodo', 'ubi', 'super', 'iam', 'tam', 'hec', 'post', 'quasi', 'ergo', 'inde', 'e', 'tunc',
                        'atque', 'ac', 'sine', 'nisi', 'nunc', 'quando', 'ne', 'usque', 'siue', 'aut', 'igitur', 'circa',
                        'quidem', 'supra', 'ante', 'adhuc', 'seu' , 'apud', 'olim', 'statim', 'satis', 'ob', 'quoniam',
                        'postea', 'nunquam']


# da provare = ['giovanniboccaccio - epistola17','giovanniboccaccio - epistola23','giovanniboccaccio - epistola8']
# ['misc - epistola21', 'misc - epistola31', 'misc - epistola20', 'misc - epistola4', 'misc - epistola1', 'misc - epistola25', 'misc - epistola2', 'misc - epistola24', 'misc - epistola10']

def load_dataset(path ='src/data/Quaestio-corpus', debug_mode=DEBUG_MODE):
    print('Loading data.\n')

    documents, authors, filenames = load_corpus(path=path, remove_epistles=False, remove_test=REMOVE_TEST, remove_egloghe=REMOVE_EGLOGHE)

    if debug_mode:
        documents, authors, filenames = documents[:100], authors[:100], filenames[:100]
        print(filenames)

    print('Data loaded.\n')

    # if remove_test:
    #     documents, authors, filenames = documents[:-1], authors[:-1], filenames[:-1]
    print([filename for filename in filenames if 'dante' in filename.lower()])

    # print('Data cleaning.\n')
    # documents = [remove_citations(doc) for doc in documents]

    return documents, authors, filenames


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
    y_test = [int(ylabel)]
    X_dev = list(np.delete(X, i))
    y_dev = list(np.delete(y, i))
    groups_dev = list(np.delete(filenames, i))
    return X_dev, X_test, y_dev, y_test, groups_dev, [doc_name] # doc_name==groups_test


def segment_data(X_dev, X_test, y_dev, y_test, groups_dev, groups_test):
    print('Data Segmentation.\n')

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


def get_processed_documents(documents, authors, filenames, processed=PROCESSED, cache_file='/home/martinaleo/.ssh/Quaestio_AV/.cache/processed_docs_def.pkl'): 

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


def extract_feature_vectors(processed_docs_dev, processed_docs_test, y_dev, groups_dev, test_document=None, oversample=OVERSAMPLE, extract_embeddings=EXTRACT_EMBEDDINGS):    #(documents, authors, filenames, nlp, target):

    print('Extracting feature vectors.')

    function_words_vectorizer = FeaturesFunctionWords(function_words=latin_function_words, ngram_range=(1,3))
    mendenhall_vectorizer = FeaturesMendenhall(upto=20)
    words_masker_SA = FeaturesDistortedView(function_words=latin_function_words, method='DVSA')
    words_masker_MA = FeaturesDistortedView(function_words=latin_function_words, method='DVMA')
    words_masker_EX = FeaturesDistortedView(function_words=latin_function_words, method='DVEX')
    sentence_len_extractor = FeaturesSentenceLength()
    POS_vectorizer = FeaturesPOST(n=(1,3))
    DEP_vectorizer = FeaturesDEP(n=(1,3))
    punct_vectorizer = FeaturesPunctuation(ngram_range=(1,1))
    char_extractor = FeaturesCharNGram(n=(2,3)) #2,3
    syllabic_quant_extractor = FeaturesSyllabicQuantities()


    vectorizers = [
            words_masker_MA,
            char_extractor,
            function_words_vectorizer,
            mendenhall_vectorizer,
            punct_vectorizer,
            sentence_len_extractor, 
            punct_vectorizer,
            POS_vectorizer ,
            DEP_vectorizer,
            syllabic_quant_extractor     
        ]
    
    # vectorizers_str = [str(vectorizer) for vectorizer in vectorizers] + ['Document embeddings'] 
    # vectorizers_str = list(feature_sets_dev.keys())

    feature_sets_dev = dict()
    feature_sets_test = dict()
    #feature_sets_test_frag = []


    for vectorizer in vectorizers:
        #extractor =  FeatureSetReductor(vectorizer)
        print('\nExtracting',vectorizer)

        reductor = FeatureSetReductor(vectorizer, k_ratio=K_RATIO)

        print('\nProcessing development set')
        features_set_dev = reductor.fit_transform(processed_docs_dev, y_dev)
        #feature_sets_dev.append(features_set_dev)

        print('\nProcessing test set')
        features_set_test = reductor.transform(processed_docs_test)
        #feature_sets_test.append(features_set_test)

        #print('\nProcessing test set segments')
        #features_test_frag = reductor.transform(processed_docs_test_frag)
        #feature_sets_test_frag.append(features_test_frag)

        if oversample:
            features_set_dev_oversampled, y_dev_oversampled, features_set_test_oversampled, groups_dev_oversampled = reductor.oversample_DRO(Xtr=features_set_dev, ytr=y_dev, Xte=features_set_test, groups=groups_dev)
            feature_sets_dev[str(vectorizer)] = features_set_dev_oversampled
            feature_sets_test[str(vectorizer)] = features_set_test_oversampled
            
        else:
            feature_sets_dev[str(vectorizer)] = features_set_dev
            feature_sets_test[str(vectorizer)] = features_set_test
        #feature_sets_test_frag.append(features_test_frag)

    if extract_embeddings:

        document_embeddings = load_embeddings('/home/martinaleo/.ssh/Quaestio_AV/src/data/embeddings/document_embeddings_segmented_docs.pkl', reshape=True, remove_test=REMOVE_TEST, remove_egloghe=REMOVE_EGLOGHE)
        document_embeddings_test = np.array(document_embeddings[test_document[0][:-2]])
        document_embeddings_test = document_embeddings_test.reshape(1, -1)

        
        docs_to_remove = [doc for doc in list(document_embeddings.keys()) if doc.startswith(test_document[0].split('_')[0] + '_')]
        for doc in docs_to_remove:
            document_embeddings.pop(doc)
        
        document_embeddings_dev = document_embeddings
        document_embeddings_dev = np.array(list(document_embeddings_dev.values()), dtype=np.float64)
        #document_embeddings_dev = list(document_embeddings_dev.values())

        if oversample:
            dro = DistributionalRandomOversampling(rebalance_ratio=0.2)
            samples = dro._samples_to_match_ratio(np.array(y_dev))
            original_indices = dro.get_original_indices(document_embeddings_dev, samples)
            document_embeddings_oversampled = [document_embeddings_dev[i] for i in original_indices]
            

            feature_sets_dev['Document embeddings'] = document_embeddings_oversampled
            
        else:
            feature_sets_dev['Document embeddings'] = document_embeddings_oversampled
            
        feature_sets_test['Document embeddings'] =  document_embeddings_test
        


    print('\nFeature vectors extracted.\n')

    if oversample:
        y_dev = y_dev_oversampled
        groups_dev = groups_dev_oversampled

    vectorizers_str = list(feature_sets_dev.keys())

    return feature_sets_dev, feature_sets_test, y_dev, groups_dev, vectorizers_str




def build_model(target, ablation_strategy='forward', save_results=True):

    hour = '0' + str(time.localtime()[3]) if len(str(time.localtime()[3])) == 1 else str(time.localtime()[3])
    minutes = '0' + str(time.localtime()[4]) if len(str(time.localtime()[4])) == 1 else str(time.localtime()[4])

    print('Start time:', hour + ':'+ minutes, '\n')
    print('Building LOO model for author', target + '.\n')

    start_time = time.time()

    documents, authors, filenames = load_dataset()
    filenames = [filename+'_0' for filename in filenames]


    processed_documents = get_processed_documents(documents, authors, filenames)
    y = [1 if author.rstrip() == target else 0 for author in authors]


    test_doc_idx = filenames.index(TEST_DOCUMENT)
    ylabel = y[test_doc_idx]
    doc = documents[test_doc_idx]
    
    np.random.seed(0)

    # LOO_split(i, X, y, doc, ylabel, filenames)
    X_dev, X_test, y_dev, y_test, groups_dev, groups_test = LOO_split(test_doc_idx, documents, y, doc, ylabel, filenames)

    X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, groups_test, groups_test_frag = segment_data(X_dev, X_test, y_dev, y_test, groups_dev, groups_test)

    X_dev_processed = get_processed_segments(processed_documents, X_dev, groups_dev, dataset='training')
    X_test_processed = get_processed_segments(processed_documents, X_test, groups_test, dataset='test')
    
    #X_test_frag_processed = get_processed_segments(processed_documents, X_test_frag, groups_test_frag, dataset='test fragments')

    feature_sets_dev, feature_sets_test, y_dev, groups_dev, vectorizers_str = extract_feature_vectors(X_dev_processed, X_test_processed, y_dev, groups_dev=groups_dev, test_document=groups_test)


    lr = [
        # (LinearSVC(random_state=RANDOM_STATE, dual='auto'), 'Linear SVC'),
        (LogisticRegression(random_state=RANDOM_STATE, n_jobs=N_JOBS), 'Logistic Regressor'),
        # (SVC(kernel='linear', probability=True, random_state=RANDOM_STATE), 'Probabilistic SVC'),
        # (CalibratedClassifierCV(estimator=LinearSVC(random_state=RANDOM_STATE, dual='auto'), cv=5, n_jobs=-1), 'Linear SVC with calibration'),
        #(AdaBoostClassifier(estimator=LinearSVC(random_state=RANDOM_STATE, dual='auto'), algorithm='SAMME', random_state=RANDOM_STATE), 'Adaboost')
    ]
    
    if ablation_strategy.lower() == 'forward':
        selector = ForwardFeatureSelector(feature_sets_dev, feature_sets_test, y_dev, y_test, groups_dev, groups_test, vectorizers_str)
        selector.forward_feature_sel()
        if save_results:
            selector.save_res(target_document=groups_test[0][:-4])


    elif ablation_strategy.lower() == 'backward':
        selector = BackwardFeatureSelector(feature_sets_dev, feature_sets_test, y_dev, y_test, groups_dev, groups_test, vectorizers_str)
        selector.backward_feature_sel()
        if save_results:
            selector.save_res(target_document=groups_test[0][:-4])

    print(f'Time spent for model building for document {(TEST_DOCUMENT[:-2])}:', round((time.time() - start_time)/60, 2), 'minutes.')

    #print(f'Time spent for model building for author {target}:', round((time.time() - start_time)/60, 2), 'minutes.')


def loop_over_authors():
    _, authors, _ = load_corpus(path='/home/martinaleo/Quaestio_AV/authorship/src/data/Quaestio-corpus')
    for author in np.unique(authors):
        if author not in ['Anonymus', 'Misc']:
            build_model(target=author)

#loop_over_authors()
            
build_model(target=TARGET, ablation_strategy='backward')

