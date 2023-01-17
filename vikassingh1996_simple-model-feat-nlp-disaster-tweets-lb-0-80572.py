import numpy as np

import pandas as pd

import time

import re

import string

import nltk

from nltk.corpus import stopwords



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
%%time

def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



# Applying the cleaning function to both test and training datasets

train['text'] = train['text'].apply(lambda x: clean_text(x))

test['text'] = test['text'].apply(lambda x: clean_text(x))
%%time

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))

test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))
%%time

def remove_stopwords(text):

    """

    Removing stopwords belonging to english language

    

    """

    words = [w for w in text if w not in stopwords.words('english')]

    return words





train['text'] = train['text'].apply(lambda x : remove_stopwords(x))

test['text'] = test['text'].apply(lambda x : remove_stopwords(x))
%%time

def combine_text(list_of_text):

    combined_text = ' '.join(list_of_text)

    return combined_text



train['text'] = train['text'].apply(lambda x : combine_text(x))

test['text'] = test['text'].apply(lambda x : combine_text(x))
lemmatizer=nltk.stem.WordNetLemmatizer()



train['text'] = train['text'].apply(lambda x: lemmatizer.lemmatize(x))

test['text'] = test['text'].apply(lambda x: lemmatizer.lemmatize(x))
X_train = train['text']

y_train = train['target']
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB 
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, ngram_range=(1, 2), norm='l2')



svc_clf = Pipeline([('tfidf', vectorizer),

                      ('svc_clf', LinearSVC())])





lr_clf = Pipeline([('tfidf', vectorizer),

                      ('lr_clf', LogisticRegression())])
svc_clf.fit(X_train,y_train)

lr_clf.fit(X_train,y_train)
test
svc_pred = svc_clf.predict(test['text'])

lr_pred = lr_clf.predict(test['text'])
submission_svc_pred=pd.DataFrame({"id":sub['id'],"target":svc_pred})

submission_lr_pred=pd.DataFrame({"id":sub['id'],"target":lr_pred})
submission_svc_pred.to_csv("submission_svc.csv",index=False)

submission_lr_pred.to_csv("submission_lr.csv",index=False)