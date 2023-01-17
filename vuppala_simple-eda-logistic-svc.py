import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
from time import time
import re
import string
import os
from pprint import pprint
import collections

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings('ignore')

np.random.seed(37)
from nltk.tokenize import TweetTokenizer
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import spacy
PATH = '../input/'
df_train = pd.read_csv(PATH + "train.tsv", sep = '\t')
df_test = pd.read_csv(PATH + "test.tsv", sep = '\t')
df_train.head(10)
sns.factorplot(x = "Sentiment", data = df_train, kind = 'count', size = 6)
df_train.Sentiment.value_counts()
print ("Number of sentences is {0:.0f}.".format(df_train.SentenceId.count()))

print ("Number of unique sentences is {0:.0f}.".format(df_train.SentenceId.nunique()))

print ("Number of phrases is {0:.0f}.".format(df_train.PhraseId.count()))
print ("The average length of phrases in the training set is {0:.0f}.".format(np.mean(df_train['Phrase'].apply(lambda x: len(x.split(" "))))))

print ("The average length of phrases in the test set is {0:.0f}.".format(np.mean(df_test['Phrase'].apply(lambda x: len(x.split(" "))))))
text = ' '.join(df_train.loc[df_train.Sentiment == 0, 'Phrase'].values)
Counter([i for i in ngrams(text.split(), 3)]).most_common(5)
print (df_train.info())
df_train.Phrase.str.len().sort_values(ascending = False)
df_train.loc[105155, 'Phrase']
df_train.Sentiment.dtype
nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])
def tokenizer(s): 
    return [w.text.lower() for w in nlp(s)]
## stemming sentences
sentences = list(df_train.Phrase.values) + list(df_test.Phrase.values)
sentences2 = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in sentences]
for i in range(len(sentences2)):sentences2[i] = ' '.join(sentences2[i])
tfidf = TfidfVectorizer(strip_accents = 'unicode', tokenizer = tokenizer, encoding='utf-8', ngram_range = (1,2), max_df = 0.75, min_df = 3, sublinear_tf = True)
_ = tfidf.fit(sentences2)
train_phrases2 = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in list(df_train.Phrase.values)]
for i in range(len(train_phrases2)):train_phrases2[i] = ' '.join(train_phrases2[i])
train_df_flags = tfidf.transform(train_phrases2)
test_phrases2 = [[stemmer.stem(word) for word in sentence.split(" ")] for sentence in list(df_test.Phrase.values)]
for i in range(len(test_phrases2)):test_phrases2[i] = ' '.join(test_phrases2[i])
test_df_flags = tfidf.transform(test_phrases2)
X_train_tf = train_df_flags[0:125000]
X_valid_tf = train_df_flags[125000:]
y_train_tf = (df_train["Sentiment"])[0:125000]
y_valid_tf = (df_train["Sentiment"])[125000:]

print("X_train shape: ", X_train_tf.shape)
print("X_valid shape: ",X_valid_tf.shape)
print("Y_train shape: ",len(y_train_tf))
print("Y_valid shape: ",len(y_valid_tf))
from sklearn.linear_model import LogisticRegression
scores = cross_val_score(LogisticRegression(C=4, dual=True), X_train_tf, y_train_tf, cv=5)
scores
np.mean(scores), np.std(scores)
logistic = LogisticRegression(C=4, dual=True)
ovrm = OneVsRestClassifier(logistic)
ovrm.fit(X_train_tf, y_train_tf)
scores = cross_val_score(ovrm, X_train_tf, y_train_tf, scoring='accuracy', n_jobs=-1, cv=3)
print (np.mean(scores))
print (np.std(scores))
print ("train accuracy:", ovrm.score(X_train_tf, y_train_tf ))
print ("valid accuracy:", ovrm.score(X_valid_tf, y_valid_tf))
df_test.head()
df_test_logistic = df_test.copy()[["PhraseId"]]
df_test_logistic['Sentiment'] = ovrm.predict(test_df_flags)

df_test_logistic.head()
svc = LinearSVC(dual=False)
svc.fit(X_train_tf, y_train_tf)
print ("train accuracy:", svc.score(X_train_tf, y_train_tf ))
print ("valid accuracy:", svc.score(X_valid_tf, y_valid_tf))
df_test_svm = df_test.copy()[["PhraseId"]]
df_test_svm['Sentiment'] = svc.predict(test_df_flags)
df_test_svm.head()
df_test_logistic.to_csv("submission_tfidf_logistic.csv", index = False)
