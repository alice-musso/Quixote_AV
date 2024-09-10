from tqdm import tqdm
import time

from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch

import numpy as np

import pickle

from string import punctuation
from data_loader import load_corpus
from splitting__ import Segmentation


PROCESSED = True
SEGMENT_MIN_TOKEN_SIZE = 400


def print_time(end=False):
    hour = '0' + str(time.localtime()[3]) if len(str(time.localtime()[3])) == 1 else str(time.localtime()[3])
    minutes = '0' + str(time.localtime()[4]) if len(str(time.localtime()[4])) == 1 else str(time.localtime()[4])
    time_str = 'Start' if not end else 'End'
    print(f'{time_str} time:' , hour + ':'+ minutes, '\n')



def load_dataset(path ='src/data/Quaestio-corpus', debug_mode=False):
    print('Loading data.\n')

    documents, authors, filenames = load_corpus(path=path, remove_epistles=False, remove_test=False, remove_egloghe=False)

    if debug_mode:
        print('debug mode')
        documents, authors, filenames = documents[:10], authors[:10], filenames[:10]

    print('Data loaded.\n')

    #print([filename for filename in filenames if 'dante' in filename.lower()])

    return documents, authors, filenames


def load_transformers_model(model_name="xlm-roberta-base", tokenizer=XLMRobertaTokenizer, model=XLMRobertaModel):
    print(f'Loading transformers model "{model_name}" and tokenizer')
    tokenizer = tokenizer.from_pretrained(model_name)
    model = model.from_pretrained(model_name)
    return model, tokenizer


def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

def pooling_strategy(embeddings, strategy='mean'):
    assert strategy in {'mean','max','min','sum'}, 'pooling strategy not valid'
    if strategy == 'mean':
        return np.mean(embeddings, axis=0)
    elif strategy == 'max':
        return np.max(embeddings, axis=0)
    elif strategy == 'min':
        return np.min(embeddings, axis=0)
    elif strategy == 'sum':
        return np.sum(embeddings, axis=0)
    

def get_token_embeddings_sw(model, tokenizer, text, tokens_to_keep, window_size=510, stride=256, pooling='mean', normalize=False):
    tokens = tokenizer.tokenize(text)

    token_embeddings = {token: [] for token in tokens_to_keep}

    # sliding window
    for i in range(0, len(tokens), stride):
        window_tokens = tokens[i:i+window_size]
        window_tokens = ['<s>'] + window_tokens + ['</s>'] #special tokens [CLS] e [SEP]
        input_ids = tokenizer.convert_tokens_to_ids(window_tokens)

        inputs = torch.tensor([input_ids])

        with torch.no_grad():
            outputs = model(inputs) # embeddings of window tokens

        last_hidden_states = outputs.last_hidden_state[0].numpy()

        for j, token in enumerate(window_tokens):
            if token in tokens_to_keep:
                token_embeddings[token].append(last_hidden_states[j])

    tokens_to_remove = []
    for token in token_embeddings:
        if token_embeddings[token]:  # Verifica che ci siano embeddings per questo token (lista non vuota)
          token_embeddings[token] = pooling_strategy(token_embeddings[token], pooling)
          if normalize:
            token_embeddings[token] = normalize_embedding(token_embeddings[token])
        else:
          #token_embeddings[token] = np.zeros(last_hidden_states.shape[-1])  # vettore di zeri
          tokens_to_remove.append(token)

    for token in tokens_to_remove:      
      token_embeddings.pop(token)

    return token_embeddings


def get_document_embeddings(filenames, token_embeddings, pooling='mean', normalize=True):
    document_embeddings = dict()
    for filename, embedding_dict in tqdm(zip(filenames, token_embeddings), total=len(filenames), desc='Computing document embeddings'):#zip(filenames, token_embeddings):

      print('Filename:', filename)

      document_embedding = pooling_strategy(list(embedding_dict.values()), pooling)
      if normalize:
          document_embedding = normalize_embedding(document_embedding)
      document_embeddings[filename] = document_embedding
    return document_embeddings


def save_embeddings(embeddings, file_name='embedding_output'):

    print('Saving embeddings')
    file_path = f'/home/martinaleo/Quaestio_AV/authorship/src/data/document_embeddings/{file_name}.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
        print(f"Embeddings saved in file {file_name}.pkl \nPath: {file_path}")


def load_embeddings(path, reshape=False, remove_test=True):
    print(f'Loading embeddings from {path}\n')
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f'Embeddings loaded.')

    if reshape:
        embedding_list = [np.array(emb).reshape(1, -1) for emb in embeddings.values()]
        embeddings_matrix = np.vstack(embedding_list)
        #print(embeddings_matrix.shape)
        embeddings_matrix_dict = {key:value for (key,value) in zip(list(embeddings.keys()), embeddings_matrix)}
        #print(list(embeddings_matrix_dict.keys())[:50])
        #print(embeddings_matrix_dict['Dante - epistola3_0'])

        if remove_test:
            test_document = 'Dante - Quaestio_0'
            docs_to_remove = [doc for doc in list(embeddings_matrix_dict.keys()) if test_document.split('_')[0] in doc]
            for doc in docs_to_remove:
                embeddings_matrix_dict.pop(doc)
        #print(embeddings_matrix_dict['Dante - epistola1_0'])
        return embeddings_matrix_dict
    else:
        return embeddings


def extract_embeddings_from_latin():
    latin_function_words = ['et',  'in',  'de',  'ad',  'non',  'ut', 'cum', 'per', 'a', 'sed', 'que', 'quia', 'ex', 'sic',
                        'si', 'etiam', 'idest', 'nam', 'unde', 'ab', 'uel', 'sicut', 'ita', 'enim', 'scilicet', 'nec',
                        'pro', 'autem', 'ibi',  'dum', 'uero', 'tamen', 'inter', 'ideo', 'propter', 'contra', 'sub',
                        'quomodo', 'ubi', 'super', 'iam', 'tam', 'hec', 'post', 'quasi', 'ergo', 'inde', 'e', 'tunc',
                        'atque', 'ac', 'sine', 'nisi', 'nunc', 'quando', 'ne', 'usque', 'siue', 'aut', 'igitur', 'circa',
                        'quidem', 'supra', 'ante', 'adhuc', 'seu' , 'apud', 'olim', 'statim', 'satis', 'ob', 'quoniam',
                        'postea', 'nunquam']
    latin_function_words_underscored = ['_' + token for token in latin_function_words]
    latin_function_words += latin_function_words_underscored
    punctuation_lst = [char for char in punctuation]
    tokens_to_keep = latin_function_words + punctuation_lst

    start_time = time.time()
    print_time()

    documents, authors, filenames = load_dataset('src/data/Quaestio-corpus')
    segmentator = Segmentation(split_policy='by_sentence', tokens_per_fragment=SEGMENT_MIN_TOKEN_SIZE)
    splitted_docs_X, splitted_docs_y = segmentator.fit_transform(documents=documents, authors=authors, filenames=filenames)
    filenames_seg = segmentator.groups # formato 'Dante - Egloga_II_2'

    model, tokenizer = load_transformers_model()
    embeddings = []

    for text in tqdm(splitted_docs_X, total=len(splitted_docs_X), desc='Extracting token embeddings'):
        embedding = get_token_embeddings_sw(
            model, tokenizer, text, tokens_to_keep,
            window_size=510, stride=256, pooling='mean', normalize=True
        )
        embeddings.append(embedding)

    save_embeddings(embeddings, file_name='token_embeddings_segmented_docs')

    document_embeddings = get_document_embeddings(filenames_seg, embeddings)

    save_embeddings(document_embeddings, file_name='document_embeddings_segmented_docs')

    print_time(end=True)
    print(f'Time spent for embeddings extraction:', round((time.time() - start_time)/60, 2), 'minutes.')


# if PROCESSED:
#     extract_embeddings_from_latin()
# else:
#     document_embeddings = load_embeddings('/home/martinaleo/Quaestio_AV/authorship/src/data/document_embeddings/document_embeddings.pkl')
#extract_embeddings_from_latin()    
#load_embeddings('/home/martinaleo/Quaestio_AV/authorship/src/data/document_embeddings/document_embeddings_segmented_docs.pkl', reshape=True)