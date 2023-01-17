# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

import nltk

# from tqdm import tqdm_notebook as tqdm

import re

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import TweetTokenizer

from sklearn.linear_model import LogisticRegression

# tqdm().pandas

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
BASE_PATH = '../input/nlp-getting-started/'

df_train = pd.read_csv(BASE_PATH+'train.csv')

df_test = pd.read_csv(BASE_PATH+'test.csv')

df_submission = pd.read_csv(BASE_PATH+'sample_submission.csv')

print(f'''df_train shape = {df_train.shape}

df_test shape = {df_test.shape}

df_submission = {df_submission.shape}

''')
df_train[df_train.target==1].head()
df_train[df_train.target==0].head()
sns.countplot(x='target', data=df_train)
df_train['lower_case_text'] = df_train.text.map(lambda x: x.lower() if isinstance(x,str) else x)

df_test['lower_case_text'] = df_test.text.map(lambda x: x.lower() if isinstance(x,str) else x)
cloud_stopwords = set(STOPWORDS)

fig, ax = plt.subplots(1,2, figsize=(20,5))

train_word_cloud = WordCloud(background_color='white', stopwords=cloud_stopwords).generate(df_train['lower_case_text'].str.cat(sep=', '))

test_word_cloud = WordCloud(background_color='white', stopwords=cloud_stopwords).generate(df_test['lower_case_text'].str.cat(sep=', '))

ax[0].imshow(train_word_cloud)

ax[1].imshow(test_word_cloud)

ax[0].axis('off')

ax[1].axis('off')

ax[0].set_title('Train Wordcoud')

ax[1].set_title('Test Wordcoud')
tokenizer = TweetTokenizer().tokenize

vect_tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', ngram_range=(1,3), max_df=0.3)

vect_tfidf.fit(df_train['lower_case_text'].values)

train_tfidf = vect_tfidf.transform(df_train['lower_case_text'].values)

test_tfidf = vect_tfidf.transform(df_test['lower_case_text'].values)
print(f'''train_tfidf shape = {train_tfidf.shape}

train_tfidf shape = {train_tfidf.shape}

''')

targets = df_train.target.values.copy()
logreg = LogisticRegression(n_jobs=-1,C= 10, class_weight= 'balanced', penalty= 'l2', random_state= 42)

val_scores = cross_val_score(estimator=logreg, X=train_tfidf, y=targets, scoring='accuracy', cv=10)

print(f'Cross_Val\nScore:{np.mean(val_scores)} +/- {np.std(val_scores)}')
# param_grid = {'penalty':['l2','l1'],

#               'C':[0.001, 0.01, 0.1, 10, 100],

#               'class_weight':['balanced', None],

#               'random_state':[42, 100, 1994]

#              }

# gridsearch = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=10, n_jobs=-1, scoring='accuracy', verbose=10)

# gridsearch.fit(train_tfidf, targets)

# print(gridsearch.best_params_)
logreg.fit(train_tfidf, targets)

preds = logreg.predict_proba(test_tfidf)
df_submission.head(10)
df_submission['target'] = np.argmax(preds,axis=1)

df_submission.to_csv('submission.csv', index=False)