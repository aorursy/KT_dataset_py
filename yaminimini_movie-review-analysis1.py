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
import pandas as pd
train = pd.read_csv("../input/train.tsv",sep = "\t")
train.head()
test = pd.read_csv("../input/test.tsv",sep = "\t")
test.head()
train["Sentiment"].unique()
train.shape
test.shape
train.isnull().sum(axis=0)
test.isnull().sum(axis = 0)
train["SentenceId"].value_counts()[0:5]
test["SentenceId"].value_counts()[0:5]
len(train["SentenceId"].unique())+len(test["SentenceId"].unique())
len(train["PhraseId"].unique())+len(test["PhraseId"].unique())
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(analyzer = "word", stop_words = 'english', min_df=0.01,max_df=0.9,ngram_range = (1,3))
tfidf.fit(train["Phrase"])
train_tfidf = tfidf.transform(train["Phrase"])
X_train = train_tfidf.todense()
Y_train = train["Sentiment"]
from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X_train,Y_train)
test_tfidf = tfidf.transform(test["Phrase"])
x_test = test_tfidf.todense()
x_test.shape
y_pred = NB.predict(x_test)
type(y_pred)
y_pred_df = pd.DataFrame(y_pred, columns = ["Sentiment"])
y_pred_df.head()
sub = pd.concat([test["PhraseId"],y_pred_df],axis = 1)
sub.head()
#sub.to_csv('submission.csv', index=False)
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
pattern = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = pattern.tokenize)
cv.fit(train['Phrase'])
train_cv = cv.transform(train["Phrase"])
train_cv
X_train2 = train_cv.todense()
Y_train2 = train["Sentiment"]
from sklearn.linear_model import SGDClassifier
sv = SGDClassifier()
##sv.fit(X_train2,Y_train2)
from sklearn.linear_model import SGDClassifier
sv = SGDClassifier(max_iter = 200)
sv.fit(X_train,Y_train)
y_pred2 = sv.predict(x_test)
y_pred2_df = pd.DataFrame(y_pred2, columns = ["Sentiment"])
sub2 = pd.concat([test["PhraseId"],y_pred2_df],axis = 1)
sub2.head()
#sub2.to_csv('submission2.csv', index=False)
from keras.preprocessing.text import Tokenizer
X_train = train['Phrase']
train.dtypes
from keras.utils import to_categorical
Y_train = to_categorical(train['Sentiment'].values)
Y_train.shape
tz = Tokenizer(num_words = 10000, lower = True)
tz.fit_on_texts(list(X_train))
X_train2 = tz.texts_to_sequences(X_train)
type(X_train2)
len(X_train2)
from keras.preprocessing.sequence import pad_sequences
X_train2 = pad_sequences(X_train2, maxlen=100)
X_train2.shape
X_test2 = tz.texts_to_sequences(test["Phrase"])
X_test2 = pad_sequences(X_test2, maxlen=100)
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train2, Y_train, test_size=0.20)
from keras.layers import Dense,Dropout,Embedding,LSTM
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential
model = Sequential()
model.add(Embedding(10000,100,mask_zero=True))
model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(5,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.fit(X_train, Y_train,validation_data = (X_val,Y_val), epochs = 4, batch_size = 32)
sub2['Sentiment'] = model.predict_classes(X_test2, batch_size=32, verbose=1)
sub2.head()
sub2.to_csv('submission3.csv', index=False)
