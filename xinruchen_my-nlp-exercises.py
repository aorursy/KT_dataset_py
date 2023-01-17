
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import gensim
import nltk
import sklearn

import re
import string
from string import digits

toxic_train = pd.read_csv('../input/train.csv')
toxic_test = pd.read_csv('../input/test.csv')

toxic_train.columns
toxic_train.head()
#toxic_train['comment_text'][669]
#text = text.replace('\n', ' ')
#text = '______'
#text.lstrip('_')
#toxic_train[toxic_train['comment_text'].str.contains('_')]
#text = toxic_train['comment_text'][8379]
#text.replace('_', '')
# Cleaning the dataset
def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(lambda elem: re.sub(r'http\S+', '', elem))
    df[text_field] = df[text_field].apply(lambda elem: elem.replace('\n', ' '))
    df[text_field] = df[text_field].apply(lambda elem: elem.replace('_', ' '))
    
    remove_digits = str.maketrans('', '', digits)
    df[text_field] = df[text_field].apply(lambda elem: elem.translate(remove_digits))
    return df

clean_toxic_train = clean_text(toxic_train, 'comment_text')
clean_toxic_test = clean_text(toxic_test, 'comment_text')

# Explore Cleaned Data
clean_toxic_train.head()
#clean_toxic_train['comment_text'][0]
#clean_toxic_train.groupby('toxic').count()

# Tokenization
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

clean_toxic_train['tokens'] = clean_toxic_train['comment_text'].apply(tokenizer.tokenize)
clean_toxic_train.head()

clean_toxic_test['tokens'] = clean_toxic_test['comment_text'].apply(tokenizer.tokenize)
# Explore words and sentences
all_words = [word for tokens in clean_toxic_train['tokens'] for word in tokens]
sentence_lengths = [len(tokens) for tokens in clean_toxic_train['tokens']]
Vocab = sorted(list(set(all_words)))
print('%s words total, with a vocabulary size of %s' %(len(all_words), len(Vocab)))
# Explore vocabulary
print(Vocab[0:100])

from collections import Counter
count_all_words = Counter(all_words)
count_all_words.most_common(100)
# Train Test Split
from sklearn.model_selection import train_test_split

list_corpus = clean_toxic_train['comment_text']
list_labels = clean_toxic_train['toxic']
#, 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

X_train, X_val, y_train, y_val = train_test_split(list_corpus, list_labels, test_size = 0.2,
                                                   random_state = 0)


#Embedding
# TFIDF bag of words
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(analyzer = 'word', token_pattern = r'\w+')

tfidf = dict()
#tfidf['train'] = (tfidf_vectorizer.fit_transform(X_train), y_train)
#tfidf['val'] = (tfidf_vectorizer.transform(X_val), y_val)

#print(tfidf['train'][0].shape)
#print(tfidf['val'][0].shape)

tfidf['all_train'] = (tfidf_vectorizer.fit_transform(clean_toxic_train['comment_text']))
tfidf['test'] = (tfidf_vectorizer.transform(clean_toxic_test['comment_text']))
print(tfidf['all_train'].shape)
print(tfidf['test'].shape)
# The Classifiers
# Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
#lr_classifier = LogisticRegression()
#lr_classifier.fit(*tfidf['train'])
#y_lr_predict = lr_classifier.predict(tfidf['val'][0])

#from sklearn.metrics import f1_score
#f1 = f1_score(tfidf['val'][1], y_lr_predict)

submission = pd.DataFrame.from_dict({'id': clean_toxic_test['id']})
label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for label_name in label_names:
    train_target = clean_toxic_train[label_name]
    classifier = LogisticRegression()
    classifier.fit(tfidf['all_train'], train_target)
    submission[label_name] = classifier.predict_proba(tfidf['test'])[:, 1]
    
submission.to_csv('submission.csv', index = False)
    
    