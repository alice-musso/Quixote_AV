from tqdm import tqdm

from data_loader import load_spanish_corpus

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


import re

from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from helpers import get_function_words

from features import (
    FeaturesFunctionWords,
    FeaturesDVEX,
    FeaturesMendenhall,
    FeaturesSentenceLength,
    FeaturesPOST,
    HstackFeatureSet, FeatureSetReductor
)

import sklearn

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression

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
    RocCurveDisplay
)

# from scikitplot.metrics import plot_roc
from sklearn.calibration import CalibratedClassifierCV

# from imblearn.over_sampling import SMOTE


# %% [markdown]
# # Data Loading

# %%
documents, authors, filenames = load_spanish_corpus(path='../../AvellanedaCorpus')

# %%
print(filenames)

# %% [markdown]
# ### Storing in DataFrame

# %%
data = data = {
    'Text': documents,
    'Author': [author.strip() for author in authors]
}

df = pd.DataFrame(columns=['Text', 'Author'], data=data)

# %%
print(df.loc[18, 'Text'])

# # Data Cleaning

# ### Noise elimination

# %%
chars = 0
for doc in df['Text']:
    chars += len(doc)
print(f'Characters before cleaning: {chars}')


# %%
def text_cleaner(text, lowercase=True):

    if lowercase:
        text = text.lower()

    text = re.sub(r'\s+\d*\s+$', '', text)
    text = re.sub(r'<.*?>', '', text) # murkup symbols
    text = re.sub(r'\*+?', '', text)
    text = re.sub(r'[\n\r_\t]', ' ', text)
    text = re.sub(r'(^libro|^acto|^paragràfo)\s[\dA-Z]*', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text=text.strip()

    return text

# %%
df['Text'] = df['Text'].apply(text_cleaner)

# %%
chars = 0
for doc in df['Text']:
    chars += len(doc)
print(f'Characters after cleaning: {chars}')

# %%
print(f'Total characters deleted: {7885585-7822318}')

# %% [markdown]
# # Segmentation & Splitting

# %%
df['Author'].value_counts()

# - we identify the sentences that make up the text if a sentence is shorter than eight words, then we merge it with the next sentence (or the previous sentence, if it is the last sentence of the text);
# - we create sequences of three consecutive sentences (hereafter: “segments”), consider each of these se- quences as a labelled text, and assign it the author label of the text from which it was extracted.

# %%
def text_segmentation(df, col_text='Text', col_author='Author', language='spanish'):
    df = df[[col_text, col_author]]
    df_split = pd.DataFrame(columns=['Text', 'Author'])
   
    for index, row in tqdm(df.iterrows()):
        author = row[col_author]
        print(author)
        doc = row[col_text]
        sentences =  sent_tokenize(doc, language=language)

        # merging short sentences 
        print('document', index,'original len:', len(sentences))
        i=1
        while i < len(sentences):
            tokens = word_tokenize(sentences[i], language=language)
            if len(tokens) < 8:
                if i < len(sentences) - 1:
                    sentences[i] = ''.join([sentences[i], sentences[i + 1]])
                    del sentences[i + 1]
                else:
                    sentences[i - 1] = ''.join([sentences[i - 1], sentences[i]])
                    del sentences[i]
            else:
                i += 1
        
        print('after first merging:', len(sentences))

        # merging 3 sentences into one single element
        for i in range(0, len(sentences), 3):
            if i < len(sentences) - 2:
                sentences[i] = ''.join(sentences[i:i+3])
                del sentences[i+1:i+3]
            elif i == len(sentences) - 2:
                sentences[i] = ''.join(sentences[i:i+1])
                del sentences[i+1:i+1]
            elif i == len(sentences) - 1:
                sentences[i-1] = ''.join(sentences[i-1:i])
                del sentences[i]
        print('after second merging:', len(sentences))
        print()

        # storing in dataframe with author label
        df_tmp = pd.DataFrame(columns=['Text', 'Author'])
        for idx, sent in enumerate(sentences):
            df_tmp.loc[idx, 'Text'] = sent
            df_tmp.loc[idx, 'Author'] = author
        df_split = pd.concat([df_split, df_tmp])

    return df_split

df_split = text_segmentation(df)
# %%
df_split['Author'].unique() == df['Author'].unique()

# %%
count=0
for col in list(df_split.columns):
    if df_split[col].isna().any():
        print(col, df_split[col].isna().sum())
        count+=1
if count==0:
    print('No NaN values found')

# %%
X = df_split['Text'].values
y = df_split['Author'].values

# %%
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# %% [markdown]
# # Data Preparation

# %% [markdown]
# ## Horizontal stacking

# %%
fuction_ws = get_function_words(lang='spanish')

# %%
# vectorizers
function_words_extractor = FeaturesFunctionWords(language='spanish')
mendenhall_extractor = FeaturesMendenhall(upto=20)
words_masker = FeatureSetReductor(FeaturesDVEX(function_words=fuction_ws))
sentence_len_extractor = FeaturesSentenceLength()
POS_extractor = FeaturesPOST(language='spanish')

# %%
hstacker = HstackFeatureSet(function_words_extractor, words_masker, mendenhall_extractor, sentence_len_extractor)
# sentence_len_extractor restituisce dei NaN
# POS_extractor non supportato per lo spagnolo
hstacked_features = hstacker.fit_transform(X_dev, authors=y_dev)

# %%
hstacked_features_test = hstacker.transform(X_test, authors=y_test)

# %% [markdown]
# # Feature selection

# %%
print(hstacked_features.shape)
print(hstacked_features_test.shape)

# %%
# feat_selector = SelectKBest(chi2, k=100)
# features_red = feat_selector.fit_transform(hstacked_features, y_dev)

# %%
# features_red_test = feat_selector.transform(hstacked_features_test)

# %%
# features_red

# %%
# features_red_test

# %% [markdown]
# ## Normalization

import sys; sys.exit()

# %%
scaler = StandardScaler(with_mean=False)
features_norm = scaler.fit_transform(features_red)

# %%
features_norm_test = scaler.transform(features_red_test)

# %% [markdown]
# # Model Building

# %%
X_dev = features_norm
X_test = features_norm_test

# %%
clf = LinearSVC(random_state=42, dual='auto')
scores = cross_val_score(clf, X_dev, y_dev, cv=5)
scores.mean(), scores.std()

# %%
clf = LogisticRegression(random_state=42, max_iter=100000)
scores = cross_val_score(clf, X_dev, y_dev, cv=5)
scores.mean(), scores.std()

# %%
#clf = SVC(random_state=42)
#scores = cross_val_score(clf, X_dev, y_dev, cv=5)
#(scores.mean(), scores.std())

# %% [markdown]
# # Model Evaluation

# %%
clf.fit(X_dev, y_dev)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')

print('Accuracy:', acc) 
print('f1:',f1)

# %%
print(classification_report(y_test, y_pred))

# %%
plt.subplots(figsize=(13,10))
cf = confusion_matrix(y_test, y_pred)
sns.heatmap(cf, annot=True, fmt='.2f', cmap="Blues", xticklabels=df['Author'].unique(), yticklabels=df['Author'].unique())
plt.xlabel("True")
plt.ylabel("Predicted")
plt.show()

# %%
clf = CalibratedClassifierCV(clf) 
clf.fit(X_dev, y_dev)
y_proba = clf.predict_proba(X_test)

# %%
plot_roc(y_test, clf.predict_proba(X_test))
plt.legend(loc='best', bbox_to_anchor=(1, 0.9), fancybox=True, shadow=True)
plt.show()
print(roc_auc_score(y_test, y_proba, multi_class="ovr", average="micro"))

# %%
df_res = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
sns.countplot(x="variable", hue="value", data=pd.melt(df_res))
plt.title('Actual distribution vs predicted', fontsize=14)

plt.legend(loc='best', bbox_to_anchor=(1, 0.9), fancybox=True, shadow=True)
plt.show()

# %% [markdown]
# # Binary classification task

# %% [markdown]
# ## Preprocessing

# %% [markdown]
# ### Binarizing target label

# %%
df['Author'].value_counts()

# %%
df_split['Author_bin'] = df_split['Author'].apply(lambda x: x if x == 'Cervantes' else 'Not_Cervantes')

# %%
df_split.head()

# %%
df_split['Author_bin'].value_counts()

# %% [markdown]
# ### Partitioning

# %%
X = df_split['Text'].values
y_bin =df_split['Author_bin'].values

# %%
X_dev, X_test, y_dev_bin, y_test_bin = train_test_split(
    X, y_bin, test_size=0.3, stratify=y_bin, random_state=42
)

# %% [markdown]
# ### Features extraction

# %%
hstacker = HstackFeatureSet(function_words_extractor, words_masker, mendenhall_extractor)
hstacked_features = hstacker.fit_transform(X_dev, authors=y_dev_bin)
hstacked_features_test = hstacker.transform(X_test, authors=y_test_bin)

# %% [markdown]
# ### Features selection

# %%
feat_selector = SelectKBest(chi2, k=100)
features_red = feat_selector.fit_transform(hstacked_features, y_dev)
features_red_test = feat_selector.transform(hstacked_features_test)

# %%
scaler = StandardScaler(with_mean=False)
features_norm = scaler.fit_transform(features_red)
features_norm_test = scaler.transform(features_red_test)

# %%
X_dev = features_norm
X_test = features_norm_test

# %% [markdown]
# ### Oversampling

# %%
sm = SMOTE(random_state=42)
X_dev, y_dev_bin = sm.fit_resample(X_dev, y_dev_bin)
np.unique(y_dev_bin, return_counts=True)

# %% [markdown]
# ### Model Building

# %%
clf = LinearSVC(random_state=42, dual='auto')
scores = cross_val_score(clf, X_dev, y_dev_bin, cv=5)
scores.mean(), scores.std()

# %%
clf_lr = LogisticRegression(random_state=42, max_iter=100000)
scores = cross_val_score(clf_lr, X_dev, y_dev_bin, cv=5)
scores.mean(), scores.std()

# %% [markdown]
# ### Model Evaluation

# %%
clf.fit(X_dev, y_dev_bin)
y_pred_bin = clf.predict(X_test)
acc = accuracy_score(y_test_bin, y_pred_bin)
f1 = f1_score(y_test_bin, y_pred_bin, pos_label='Cervantes')

print('Accuracy:', acc) 
print('f1:',f1)

# %%
print(classification_report(y_test_bin, y_pred_bin))

# %%
plt.subplots(figsize=(13,10))
cf = confusion_matrix(y_test_bin, y_pred_bin)
sns.heatmap(cf, annot=True, fmt='.2f', cmap="Blues", xticklabels=df_split['Author_bin'].unique(), yticklabels=df_split['Author_bin'].unique())
plt.xlabel("True")
plt.ylabel("Predicted")
plt.show()

# %%
clf = CalibratedClassifierCV(clf) 
clf.fit(X_dev, y_dev_bin)
y_proba_bin = clf.predict_proba(X_test)
y_proba_bin = [y[1] for y in y_proba_bin]

plot_roc(y_test_bin, clf.predict_proba(X_test))
plt.legend(loc='best', bbox_to_anchor=(1, 0.9), fancybox=True, shadow=True)
plt.show()
print(roc_auc_score(y_test_bin, y_proba_bin))

# %%
df_res = pd.DataFrame({'Actual': y_test_bin, 'Predicted': y_pred_bin})
sns.countplot(x="variable", hue="value", data=pd.melt(df_res))
plt.title('Actual distribution vs predicted', fontsize=14)

plt.legend(loc='best', bbox_to_anchor=(1, 0.9), fancybox=True, shadow=True)
plt.show()



if __name__ == '__main__':
    print("hola")