import os
import sys
from os.path import join
import collections
from glob import glob, escape
from pathlib import Path
from itertools import chain
import numpy as np
from tqdm import tqdm
import re
from collections import Counter

# ------------------------------------------------------------------------
# document loading routine
# ------------------------------------------------------------------------

def get_latin_function_words():
    return ['et',  'in',  'de',  'ad',  'non',  'ut', 'cum', 'per', 'a', 'sed', 'que', 'quia', 'ex', 'sic',
                        'si', 'etiam', 'idest', 'nam', 'unde', 'ab', 'uel', 'sicut', 'ita', 'enim', 'scilicet', 'nec',
                        'pro', 'autem', 'ibi',  'dum', 'uero', 'tamen', 'inter', 'ideo', 'propter', 'contra', 'sub',
                        'quomodo', 'ubi', 'super', 'iam', 'tam', 'hec', 'post', 'quasi', 'ergo', 'inde', 'e', 'tunc',
                        'atque', 'ac', 'sine', 'nisi', 'nunc', 'quando', 'ne', 'usque', 'siue', 'aut', 'igitur', 'circa',
                        'quidem', 'supra', 'ante', 'adhuc', 'seu' , 'apud', 'olim', 'statim', 'satis', 'ob', 'quoniam',
                        'postea', 'nunquam']
    
def get_latin_verbal_endings():
    return ['o', 'eo', 'io', 'as', 'es', 'is', 'at', 'et', 'amus', 'emus', 'atis',
                'etis', 'itis', 'ant', 'ent', 'unt', 'iunt', 'or', 'eor', 'ior', 'aris', 'eris', 'iris',
                'atur', 'etur', 'itur', 'amur', 'emur', 'imur', 'amini', 'emini', 'imini',
                'antur', 'entur', 'untur', 'iuntur', 'abam', 'ebam', 'iebam', 'abas',
                'ebas', 'iebas', 'abat', 'ebat', 'iebat', 'abamus', 'ebamus', 'iebamus',
                'abatis', 'ebatis', 'iebatis', 'abant', 'ebant', 'iebant', 'abar', 'ebar',
                'iebar', 'abaris', 'ebaris', 'iebaris', 'abatur', 'ebatur', 'iebatur',
                'abamur', 'ebamur', 'iebamur', 'abamini', 'ebamini', 'iebamini',
                'abantur', 'ebantur', 'iebantur', 'abo', 'ebo', 'am', 'iam', 'abis', 'ebis',
                'ies', 'abit', 'ebit', 'iet', 'abimus', 'ebimus', 'iemus', 'abitis',
                'ebitis', 'ietis', 'abunt', 'ebunt', 'ient', 'abor', 'ebor', 'ar', 'iar',
                'aberis', 'eberis', 'ieris', 'abitur', 'ebitur', 'ietur', 'abimur', 'ebimur',
                'iemur', 'abimini', 'ebimini', 'iemini', 'abuntur', 'ebuntur', 'ientur',
                'i', 'isti', 'it', 'imus', 'istis', 'erunt', 'em', 'eam', 'eas', 'ias', 'eat', 'iat',
                'eamus', 'iamus', 'eatis', 'iatis', 'eant', 'iant', 'er', 'ear', 'earis', 'iaris',
                'eatur', 'iatur', 'eamur', 'iamur', 'eamini', 'iamini', 'eantur', 'iantur',
                'rem', 'res', 'ret', 'remus', 'retis', 'rent', 'rer', 'reris', 'retur', 'remur',
                'remini', 'rentur', 'erim', 'issem', 'isses', 'isset', 'issemus', 'issetis',
                'issent', 'a', 'ate', 'e', 'ete', 'ite', 'are', 'ere', 'ire', 'ato', 'eto', 'ito',
                'atote', 'etote', 'itote', 'anto', 'ento', 'unto', 'iunto', 'ator', 'etor',
                'itor', 'aminor', 'eminor', 'iminor', 'antor', 'entor', 'untor', 'iuntor',
                'ari', 'eri', 'iri', 'andi', 'ando', 'andum', 'andus', 'ande', 'ans', 'antis',
                'anti', 'antem', 'antes', 'antium', 'antibus', 'antia', 'esse', 'sum',
                'est', 'sumus', 'estis', 'sunt', 'eram', 'eras', 'erat', 'eramus', 'eratis',
                'erant', 'ero', 'erit', 'erimus', 'eritis', 'erint', 'sim', 'sis', 'sit',
                'simus', 'sitis', 'sint', 'essem', 'esses', 'esset', 'essemus', 'essetis',
                'essent', 'fui', 'fuisti', 'fuit', 'fuimus', 'fuistis', 'fuerunt', 'este', 'esto',
                'estote', 'sunto']


# data_loader.py

def load_corpus(path: str, **filters) -> tuple[list[str], list[str], list[str]]:
    """Load corpus documents with optional filtering.
    
    Args:
        path: Directory path containing corpus files
        **filters: Boolean flags for filtering:
            - remove_epistles: Remove epistolary texts
            - remove_test: Remove test document (Quaestio)
            - remove_egloghe: Remove eclogues 
            - remove_anonymus_files: Remove anonymous/misc texts
            - remove_unique_authors: Remove texts by authors with single work
            - remove_monarchia: Remove Monarchia text
    
    Returns:
        Tuple of (documents, authors, filenames)
    """
    files = [f for f in Path(path).glob('*.txt')]
    corpus = []
    
    for file in tqdm(files, desc=f'Loading corpus from {path}'):
        if _should_skip_file(file.name, filters):
            print(f'Removing {file.name}')
            continue
            
        author, title = file.stem.split('-')
        text = _clean_text(file.read_text(encoding='utf8', errors='ignore'))
        
        corpus.append({
            'text': text,
            'author': author.strip(),
            'filename': file.stem
        })

    if filters.get('remove_unique_authors'):
        corpus = _remove_single_author_texts(corpus)

    documents = [doc['text'] for doc in corpus]
    authors = [doc['author'] for doc in corpus]
    filenames = [doc['filename'] for doc in corpus]

    print(f'Total documents: {len(documents)}')
    print(f'Total authors: {len(set(authors))}')
    
    return documents, authors, filenames

def _should_skip_file(filename: str, filters: dict) -> bool:
    """Check if file should be filtered out based on criteria."""
    checks = {
        'remove_epistles': lambda f: 'epistola' in f.lower(),
        'remove_test': lambda f: 'quaestio' in f.lower(),
        'remove_egloghe': lambda f: 'egloga' in f.lower(),
        'remove_anonymus_files': lambda f: any(x in f.lower() for x in ['misc', 'anonymus']),
        'remove_monarchia': lambda f: 'monarchia' in f.lower()
    }
    return any(check(filename) for flag, check in checks.items() if filters.get(flag))

def _clean_text(text: str) -> str:
    """Clean and normalize text content."""
    text = text.lower()
    text = re.sub(r'\{[^{}]*\}', '', text)
    text = re.sub(r'\*[^**]*\*', '', text) 
    text = re.sub(r'<\w>(.*?)</\w>', r'\1', text)
    text = text.replace('\x00', '')
    return text.strip()

def _remove_single_author_texts(corpus: list[dict]) -> list[dict]:
    """Remove texts by authors who only have one work."""
    author_counts = Counter(doc['author'] for doc in corpus)
    return [doc for doc in corpus if author_counts[doc['author']] > 1]



# def load_corpus__(path, remove_epistles=False, remove_test=True, remove_egloghe=False, remove_anonymus_files=False, remove_unique_authors=False, remove_monarchia=False):
#     filenames = []
#     authors = []
#     documents = []
#     ndocs=0
#     dirs = os.listdir(path)
#     for file in tqdm(dirs, total=len(dirs), desc='loading: ' + path):
#         if file.endswith('txt'):
            
#             if remove_epistles:
#                 files_removed = 0
#                 if 'epistola' in file.lower():
#                     print('removing', file)
#                     files_removed += 1
#                     continue
            
#             if remove_egloghe:

#                 if 'egloga' in file.lower():
#                     print('removing egloga', file)
#                     continue

#             if remove_test:
#                 if ' quaestio' in file.lower():
#                     print('removing test document', file)
#                     continue

#             if remove_anonymus_files:

#                 if 'misc' in file.lower():
#                     print('Removing anonymus or misc text')
#                     continue

#                 if 'anonymus' in file.lower():
#                     print('Removing anonymus or misc text')
#                     continue

#             if remove_monarchia:
#                 if 'monarchia' in file.lower():
#                         print('removing document', file)
#                         continue

#             file_name = file.replace('.txt','')
#             author, textname = file_name.split('-')
#             text = open(join(path,file), encoding= "utf8", errors='ignore').read()

#             #cleaning
#             text = re.sub('\{[^{}]*\}', "", text)
#             text = re.sub('\*[^**]*\*', "", text)
#             text=text.lower()
#             text = text.replace('\x00', '') #remove null bytes
#             text = re.sub('<\w>(.*?)</\w>', '\1', text)
#             text=text.strip()
#             #text = remove_citations(text) # da rivedere
   
#             documents.append(text)
#             filenames.append(file_name)
#             authors.append(author.strip())
#             ndocs += 1

#     if remove_unique_authors:
#         print('removing unique authors')
#         filtered_authors, indices_to_remove = remove_unique_elements_and_save_indices(authors)
#         filtered_documents = [doc for i, doc in enumerate(documents) if i not in indices_to_remove]
#         filtered_filenames = [filename for i, filename in enumerate(filenames) if i not in indices_to_remove]

#         print('Total filtered documents:', len(filtered_documents))
#         print('Total filtered authors:', len(list(set(filtered_authors))))
#         return filtered_documents, filtered_authors, filtered_filenames
        
#     print('Total documents:', len(documents))
#     print('Total authors:', len(list(set(authors))))

#     return documents, authors, filenames


# def remove_unique_elements_and_save_indices(lst):
#     # Conta le occorrenze di ciascun elemento
#     element_counts = Counter(lst)
    
#     # Trova gli indici degli elementi unici
#     unique_indices = [i for i, elem in enumerate(lst) if element_counts[elem] == 1]
    
#     # Rimuovi gli elementi unici dalla lista originale
#     lst_filtered = [elem for elem in lst if element_counts[elem] > 1]
#     print(len(lst_filtered))
    
#     return lst_filtered, unique_indices






