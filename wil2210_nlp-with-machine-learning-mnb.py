import os
import numpy as np
import pandas as pd
import spacy

import matplotlib.pyplot as plt
import seaborn as sns

sns.set
%matplotlib inline
path = '../input'
df_train = pd.read_csv(os.path.join(path, 'train.csv'))
df_test = pd.read_csv(os.path.join(path, 'test.csv'))
df_train.head()
df_train.shape, df_test.shape
df_train.info()
df_train['target'].value_counts().plot(kind='bar');
insincere_ratio = (80810 / 1225312) * 100
insincere_ratio
y = df_train['target']
X = df_train['question_text']
X_insincere = X[y == 1]
X_insincere.head()
X_sincere = X[y == 0]
X_sincere.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
def tokenize(data):
    corpus = [word_tokenize(token) for token in data]
    lowercase_train = [[token.lower() for token in doc] for doc in corpus]
    alphas = [[token for token in doc if token.isalpha()] for doc in lowercase_train]
    stop_words = stopwords.words('english')
    train_no_stop = [[token for token in doc if token not in stop_words] for doc in alphas]
    stemmer = PorterStemmer()
    stemmed = [[stemmer.stem(token) for token in doc] for doc in train_no_stop]
    train_clean_str = [ ' '.join(doc) for doc in stemmed]
    return train_clean_str
X_train = tokenize(X_train)
X_test = tokenize(X_test)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
tvec = TfidfVectorizer(stop_words='english')
cvec = CountVectorizer(stop_words='english')
svd = TruncatedSVD(n_components=100, random_state=42)
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
pipe = Pipeline([('vectorizer', cvec), ('mnb', mnb)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
labels = ['sincere', 'unsincere']
cm = pd.DataFrame(cm, columns=labels, index=labels)
cm
from sklearn.model_selection import cross_val_score
cv = cross_val_score(pipe, X_test, y_test, scoring='f1_macro', cv=5).mean()
cv
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)
test = df_test['question_text']
test = tokenize(test)
y_pred = pipe.predict(test)
y_pred
path = '../input'
df_sub = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
df_sub['prediction'] = y_pred
df_sub.to_csv("submission.csv", index=False)