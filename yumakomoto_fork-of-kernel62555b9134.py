# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install contractions

#!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import TweetTokenizer

import matplotlib.pyplot as plt

import seaborn as sns



import string # 記号(punctuation)の一覧を取得

import gc

import re

import operator

import contractions # 文書の短縮形を復元

from wordcloud import STOPWORDS # ストップワードのリストを取得する

from collections import defaultdict # n-gram作成時に利用

#import tokenization



import tensorflow as tf

import tensorflow_hub as hub

from tensorflow import keras

from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
train_df = pd.read_csv('../input/nlp-getting-started/train.csv')

test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
train_df['lower_case_text'] = train_df.text.map(lambda x: x.lower() if isinstance(x,str) else x)

test_df['lower_case_text'] = test_df.text.map(lambda x: x.lower() if isinstance(x,str) else x)
#各カラムの欠損値率

print("missing-value ratio of training data(%)")

print(train_df.isnull().sum()/train_df.shape[0]*100)

print("\nmissing-value ratio of test data(%)")

print(test_df.isnull().sum()/test_df.shape[0]*100)
missing_cols = ['keyword','location','text']



fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)



sns.barplot(x=train_df[missing_cols].isnull().sum().index, y=train_df[missing_cols].isnull().sum().values, ax=axes[0])

sns.barplot(x=test_df[missing_cols].isnull().sum().index, y=test_df[missing_cols].isnull().sum().values, ax=axes[1])



axes[0].set_ylabel('Missing Value Count', size=15, labelpad=20)

axes[0].tick_params(axis='x', labelsize=15)

axes[0].tick_params(axis='y', labelsize=15)

axes[1].tick_params(axis='x', labelsize=15)

axes[1].tick_params(axis='y', labelsize=15)



axes[0].set_title('Training Set', fontsize=13)

axes[1].set_title('Test Set', fontsize=13)



plt.show()



for df in [train_df, test_df]:

    for col in ['keyword', 'location','text']:

        df[col] = df[col].fillna(f'no_{col}')
train_df['target_mean'] = train_df.groupby('keyword')['target'].transform('mean')



fig = plt.figure(figsize=(8, 72), dpi=100)



sns.countplot(y=train_df.sort_values(by='target_mean', ascending=False)['keyword'],

             hue=train_df.sort_values(by='target_mean', ascending=False)['target'])



plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=12)

plt.legend(loc=1)

plt.title('Target Distribution in Keywords')



plt.show()





train_df.drop(columns=['target_mean'], inplace=True)
DISASTER_TWEETS = train_df['target'] == 1
def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [' '.join(ngram) for ngram in ngrams]



N = 100



# ユニグラム

disaster_unigrams = defaultdict(int)

nondisaster_unigrams = defaultdict(int)



for tweet in train_df[DISASTER_TWEETS]['text']:

    for word in generate_ngrams(tweet):

        disaster_unigrams[word] += 1

        

for tweet in train_df[~DISASTER_TWEETS]['text']:

    for word in generate_ngrams(tweet):

        nondisaster_unigrams[word] += 1

        

df_disaster_unigrams = pd.DataFrame(sorted(disaster_unigrams.items(), key=lambda x: x[1])[::-1])

df_nondisaster_unigrams = pd.DataFrame(sorted(nondisaster_unigrams.items(), key=lambda x: x[1])[::-1])



# バイグラム

disaster_bigrams = defaultdict(int)

nondisaster_bigrams = defaultdict(int)



for tweet in train_df[DISASTER_TWEETS]['text']:

    for word in generate_ngrams(tweet, n_gram=2):

        disaster_bigrams[word] += 1

        

for tweet in train_df[~DISASTER_TWEETS]['text']:

    for word in generate_ngrams(tweet, n_gram=2):

        nondisaster_bigrams[word] += 1

        

df_disaster_bigrams = pd.DataFrame(sorted(disaster_bigrams.items(), key=lambda x: x[1])[::-1])

df_nondisaster_bigrams = pd.DataFrame(sorted(nondisaster_bigrams.items(), key=lambda x: x[1])[::-1])



# トライグラム

disaster_trigrams = defaultdict(int)

nondisaster_trigrams = defaultdict(int)



for tweet in train_df[DISASTER_TWEETS]['text']:

    for word in generate_ngrams(tweet, n_gram=3):

        disaster_trigrams[word] += 1

        

for tweet in train_df[~DISASTER_TWEETS]['text']:

    for word in generate_ngrams(tweet, n_gram=3):

        nondisaster_trigrams[word] += 1

        

df_disaster_trigrams = pd.DataFrame(sorted(disaster_trigrams.items(), key=lambda x: x[1])[::-1])

df_nondisaster_trigrams = pd.DataFrame(sorted(nondisaster_trigrams.items(), key=lambda x: x[1])[::-1])
fig, axes = plt.subplots(ncols=2, figsize=(18, 50), dpi=100)

plt.tight_layout()



sns.barplot(y=df_disaster_unigrams[0].values[:N], x=df_disaster_unigrams[1].values[:N], ax=axes[0], color='red')

sns.barplot(y=df_nondisaster_unigrams[0].values[:N], x=df_nondisaster_unigrams[1].values[:N], ax=axes[1], color='green')



for i in range(2):

    axes[i].spines['right'].set_visible(False)

    axes[i].set_xlabel('')

    axes[i].set_ylabel('')

    axes[i].tick_params(axis='x', labelsize=13)

    axes[i].tick_params(axis='y', labelsize=13)



axes[0].set_title(f'Top {N} most common unigrams in Disaster Tweets', fontsize=15)

axes[1].set_title(f'Top {N} most common unigrams in Non-disaster Tweets', fontsize=15)



plt.show()
fig, axes = plt.subplots(ncols=2, figsize=(18, 50), dpi=100)

plt.tight_layout()



sns.barplot(y=df_disaster_bigrams[0].values[:N], x=df_disaster_bigrams[1].values[:N], ax=axes[0], color='red')

sns.barplot(y=df_nondisaster_bigrams[0].values[:N], x=df_nondisaster_bigrams[1].values[:N], ax=axes[1], color='green')



for i in range(2):

    axes[i].spines['right'].set_visible(False)

    axes[i].set_xlabel('')

    axes[i].set_ylabel('')

    axes[i].tick_params(axis='x', labelsize=13)

    axes[i].tick_params(axis='y', labelsize=13)



axes[0].set_title(f'Top {N} most common bigrams in Disaster Tweets', fontsize=15)

axes[1].set_title(f'Top {N} most common bigrams in Non-disaster Tweets', fontsize=15)



plt.show()
fig, axes = plt.subplots(ncols=2, figsize=(20, 50), dpi=100)



sns.barplot(y=df_disaster_trigrams[0].values[:N], x=df_disaster_trigrams[1].values[:N], ax=axes[0], color='red')

sns.barplot(y=df_nondisaster_trigrams[0].values[:N], x=df_nondisaster_trigrams[1].values[:N], ax=axes[1], color='green')



for i in range(2):

    axes[i].spines['right'].set_visible(False)

    axes[i].set_xlabel('')

    axes[i].set_ylabel('')

    axes[i].tick_params(axis='x', labelsize=13)

    axes[i].tick_params(axis='y', labelsize=11)



axes[0].set_title(f'Top {N} most common trigrams in Disaster Tweets', fontsize=15)

axes[1].set_title(f'Top {N} most common trigrams in Non-disaster Tweets', fontsize=15)



plt.show()
train_df.fillna('FILL')

test_df.fillna('FILL')


def fix_contractions(text):

    return contractions.fix(text)



# 関数適用前のツイート例

print("tweet before contractions fix : ", train_df.iloc[1055]["text"])



# 関数の適用

train_df['text']=train_df['text'].apply(lambda x : fix_contractions(x))

test_df['text']=test_df['text'].apply(lambda x : fix_contractions(x))



# 関数適用後のツイート例

print("tweet after contractions fix : ", train_df.iloc[1055]["text"])

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



# 関数適応前のツイート例

print("tweet before URL removal : ", train_df.iloc[1055]["text"])



# 関数の適用

train_df['text']=train_df['text'].apply(lambda x : remove_URL(x))

test_df['text']=test_df['text'].apply(lambda x : remove_URL(x))



# 関数適用後のツイート例

print("tweet after URL removal : ", train_df.iloc[1055]["text"])
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



# 関数適応前のツイート例

print("tweet before punctuation removal : ", train_df.iloc[1055]["text"])



# 関数の適用

train_df['text']=train_df['text'].apply(lambda x : remove_punct(x))

test_df['text']=test_df['text'].apply(lambda x : remove_punct(x))



# 関数適用後のツイート例

print("tweet after punctuation removal : ", train_df.iloc[1055]["text"])
train_df.head()
tokenizer = TweetTokenizer().tokenize

vect_tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', ngram_range=(1,3), max_df=0.3)

vect_tfidf.fit(train_df['lower_case_text'].values)
train_vectors = vect_tfidf.transform(train_df['lower_case_text'].values)

test_vectors = vect_tfidf.transform(test_df['lower_case_text'].values)
clf = LogisticRegression(n_jobs=-1,C= 10, class_weight= 'balanced', penalty= 'l2', random_state= 42)

scores = cross_val_score(clf, train_vectors,train_df["target"], cv=3, scoring="f1")

#print(f'Cross_Val\nScore:{np.mean(scores)} +/- {np.std(scores)}')

print(scores)
clf.fit(train_vectors, train_df["target"])
sub = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sub["target"] = clf.predict(test_vectors)
sub.head(20)
sub.to_csv("submission.csv", index=False)