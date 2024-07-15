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

from features import ( 
    FeaturesFunctionWords, 
    FeaturesDVEX, 
    FeaturesMendenhall, 
    FeaturesSentenceLength, 
    FeaturesPOST, 
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

from model_builder import (Fun_StackingClassifier, 
                           split_n_segment,
                           split, 
                           load_model,
                            get_scores,
                            plot_res)


import pickle
import warnings

import warnings
warnings.filterwarnings('ignore') 



# support functions



def get_group_names(groups, filenames):
    groups_names = [filenames[doc_idx] for doc_idx in groups]
    groups_names_distribution = {name:round((np.unique(groups_names, return_counts=True)[1][i])/len(groups_names),2) 
                                 for i, name in enumerate(np.unique(groups_names))}
    return groups_names, groups_names_distribution




def load_model(filename, target, folder):
    path = f'/home/martinaleo/.ssh/authorship/src/data/{folder}'
    os.chdir(path)
    #os.chdir('/home/martinaleo/.ssh/authorship/src/data/FunTAT_normalize_false_models')
    filename = f"{target}-{filename}.pickle"
    return pickle.load(open(filename, "rb"))


def find_mismatched_docs(y_test, y_pred, group_names_test, groups_names_distribution, plot_res=True):
    res_df = pd.DataFrame(columns=['Mismatched_doc', 'Distribution_over_total_errors', 'Distribution_over_test_set'])
    wrong_indices = np.where(y_test != y_pred)[0]
    mismatched_docs = [group_names_test[idx] for idx in wrong_indices]
    distribution = {name:round((np.unique(mismatched_docs, return_counts=True)[1][i])/len(mismatched_docs),2) 
                                 for i, name in enumerate(np.unique(mismatched_docs))}

    for i, (key, val) in enumerate(distribution.items()):    
        res_df.loc[i]={
            'Mismatched_doc':  key,
            'Distribution_over_total_errors': val,
            'Distribution_over_test_set':groups_names_distribution[key] 
        }

    if plot_res:
        positions = np.arange(len(res_df))
        plt.bar(positions, res_df['Distribution_over_total_errors'], 0.35, label='Distribution_over_total_errors')
        plt.bar(positions + 0.35, res_df['Distribution_over_test_set'], 0.35, label='Distribution_over_test_set')

        plt.xlabel('Mismatched docs')
        plt.ylabel('Relative distribution')
        plt.title('Comparing distribution of docs over errors and test set')
        plt.xticks(positions + 0.35 / 2, res_df['Mismatched_doc'].apply(lambda x: x.split(' - ')[1]))
        plt.xticks(rotation=90)
        plt.legend()

        plt.show()
    return mismatched_docs, distribution

# main

def run_res(model_type='tat', normalize=False, plot_res=False):
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

    # model loading
    for author in np.unique(authors):
        if author in np.unique(authors)[5:]:
            if model_type.lower() == 'single_clf':
                model_name = 'Single_clf_'
                folder_name ='SingleClf_models'
                print()
            model_name = f'Fun_{model_type.capitalize()}clf_norm_True' if normalize else f'Fun_{model_type.capitalize()}_clf_norm_False'
            folder_name = f'Fun{model_type.upper()}_normalize_true_models' if normalize else f'Fun{model_type.upper()}_normalize_false_models'
            print()
            print('Loading model', model_name, 'for author', author)
            clf=load_model(model_name, author, folder=folder_name)
            print()

            # preparing data
            X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, groups_test, groups_test_frag = split_n_segment(documents=documents, authors=authors, target=author)

            # computing res
            print('Model loaded. Computing res...')
            print('On whole docs:')
            y_pred = get_scores(clf, X_test, y_test)
            print()
            print('On fragmented docs:')
            y_pred_frag = get_scores(clf, X_test_frag, y_test_frag)
            print()

            # plotting res
            if plot_res:
                print('Plotting res...')
                plot_res(y_test, y_pred, target=author)

                # computing mismatched docs
                #goups_names_dev, groups_names_distribution_dev =get_group_names(groups_dev,filenames)
                goups_names_test, groups_names_distribution_test =get_group_names(groups_test,filenames)
                goups_names_test_frag, groups_names_distribution_test_frag = get_group_names(groups_test_frag,filenames)  

                mismatched_docs, mismatched_docs_distribution = find_mismatched_docs(y_test,
                                                                                    y_pred, 
                                                                                    goups_names_test, 
                                                                                    groups_names_distribution_test)
                
                mismatched_docs_frag, mismatched_docs_distribution_frag = find_mismatched_docs(y_test_frag,
                                                                                    y_pred_frag, 
                                                                                    goups_names_test_frag, 
                                                                                    groups_names_distribution_test_frag)

run_res()
# auths= ['Agustín de Rojas Villandrando ', 'Alonso de Castillo Solórzano ',
#         'Cervantes ', 'Cristóbal Suárez de Figueroa ',
#         'Guillén de Castro ', 'Juan Ruiz de Alarcón y Mendoza ',
#         'Lope de Vega ', 'Mateo Alemán ', 'Pasamonte ', 'Pérez de Hita ',
#         'Quevedo ', 'Tirso de Molina ']
# authors=[auth.strip() for auth in auths]


# c =5
# while c <10:
#     res=f"=("
#     for auth in authors:
#         res+=f"'{auth}'!L{c}+"
#     res_fin = res[:-1] + f')/{len(auths)}'
#     print(res_fin)
#     print()
#     c+=1



