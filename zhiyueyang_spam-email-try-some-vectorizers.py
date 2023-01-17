# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline
train_data = pd.read_csv('./datasets/train.csv', encoding='latin-1')
test_data = pd.read_csv('./datasets/test.csv', encoding='latin-1')
test_data.head()
train_data.info()
print('========================')
test_data.info()
print(train_data.describe())
print('========================')
test_data.describe(include='O')
train_data.groupby(by='v1').describe()
train_data['length'] = train_data['v2'].apply(len)
train_data.hist(column='length', by='v1', bins=50,figsize=(10,4))
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
def text_process(content):
    translator = str.maketrans('', '', string.punctuation)
    content_nopunc = content.translate(translator)
    return [word for word in content_nopunc.split() if word.lower() not in stopwords.words('english')]
all_mail = train_data['v2'].append(test_data['v2'], ignore_index=True)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer=text_process)
mail_tfidf = tfidf_vectorizer.fit_transform(all_mail)
train_tfidf = mail_tfidf[:960]
test_tfidf = mail_tfidf[960:]
from sklearn.feature_extraction.text import HashingVectorizer
hash_vectorizer = HashingVectorizer(analyzer=text_process)
mail_hash = hash_vectorizer.fit_transform(all_mail)
train_hash = mail_hash[:960]
test_hash = mail_hash[960:]
from gensim.models.word2vec import Word2Vec
corpus = list(train_data['v2'].apply(text_process)) + list(test_data['v2'].apply(text_process))
num_features = 500
min_word_count = 40
num_workers = 4
window_size = 10
downsampling = 1e-3

# train word2vec model using gensim
model = Word2Vec(corpus, window=window_size,size=num_features,
                 min_count=min_word_count,workers=num_workers,sample=downsampling)
def to_vector(content):
    words = text_process(content)
    array = np.array([model.wv.__getitem__(w) for w in words if model.wv.__contains__(w)])
    return pd.Series(array.mean(axis=0))
all_X = train_data['v2'].apply(to_vector)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC

clf1 = LogisticRegressionCV(cv=5)
clf2 = LinearSVC(C=3, random_state=5)

from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2)], voting='hard')
model.fit(train_tfidf, train_data['v1'])
holdout_predictions = model.predict(test_tfidf)
