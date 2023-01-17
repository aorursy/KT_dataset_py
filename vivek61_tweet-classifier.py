# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.

import seaborn as sns

import string

import nltk

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

import matplotlib.pyplot as plt

import re

from sklearn.model_selection import cross_val_score, train_test_split



from sklearn.linear_model import RidgeClassifier, LogisticRegression

from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer

import keras 
df = pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')
df=df[['text','target']]

test=test[['text']]
def hash_count(tweet):

    w = tweet.split()

    return len([word for word in w if word.startswith('#')])



def mention_count(tweet):

    w = tweet.split()

    return len([word for word in w if word.startswith('@')])



def avg_word_len(tweet):

    w = tweet.split()

    word_len = [len(word) for word in w]

    return sum(word_len)/len(word_len)



df['no_chars'] = df['text'].apply(len)

df['no_words'] = df['text'].str.split().apply(len)

df['no_sent'] = df['text'].str.split('.').apply(len)

df['no_para'] = df['text'].str.split('\n').apply(len)

df['avg_word_len'] = df['text'].apply(avg_word_len)

df['no_hashtags'] = df['text'].apply(hash_count)

df['no_mentions'] = df['text'].apply(mention_count)



test['no_chars'] = test['text'].apply(len)

test['no_words'] = test['text'].str.split().apply(len)

test['no_sent'] = test['text'].str.split('.').apply(len)

test['no_para'] = test['text'].str.split('\n').apply(len)

test['avg_word_len'] = test['text'].apply(avg_word_len)

test['no_hashtags'] = test['text'].apply(hash_count)

test['no_mentions'] = test['text'].apply(mention_count)

df.isna().sum()
df.head(10)
stop = set(stopwords.words('english'))



d=[]

for s in df['text']:

    



       d.append([x for x in s.split() if x not in stop])

cleant=[]

for n in d:

    cleant.append(' '.join(n))        

df['newt']=cleant







d=[]

for s in test['text']:

    d.append([x for x in s.split() if x not in stop])

cleante=[]

for n in d:

    cleante.append(' '.join(n))        

test['newt']=cleante







url=[]

for s in cleant:

    url.append(re.findall('http[s]?://.*',s))

df['url']=url

df['url_cnt']=df['url'].apply(len)

df.sample(5)



url=[]

for s in cleante:

    url.append(re.findall('http[s]?://.*',s))

test['url']=url

test['url_cnt']=test['url'].apply(len)

sns.countplot(hue='target',x='url_cnt',data=df)
from nltk import SnowballStemmer

stems=[]

clean2t=[]



# Function to apply stemming to a list of words

stemmer = SnowballStemmer(language='english')

for sen in cleant:

     stems.append([stemmer.stem(word) for word in sen.split()])

        

for n in stems:

    clean2t.append(' '.join(n))        

df['newt']=clean2t

        





stems=[]

clean2te=[]

    

for sen in cleante:

     stems.append([stemmer.stem(word) for word in sen.split()])

        

for n in stems:

    clean2te.append(' '.join(n))        

test['newt']=clean2te 

            
Xtrain=df.drop('target',axis=1)

ytrain=df['target']
vectorizer = TfidfVectorizer()

XX = vectorizer.fit(Xtrain['newt'])

X=XX.transform(Xtrain['newt'])

count_vect_df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())

train = pd.concat([Xtrain, count_vect_df], axis=1)

train=train.drop('newt',axis=1)

train=train.drop('url',axis=1)

train=train.drop('text',axis=1)



Y=XX.transform(test['newt'])

count_vect_df = pd.DataFrame(Y.todense(), columns=vectorizer.get_feature_names())

test = pd.concat([test, count_vect_df], axis=1)

test=test.drop('newt',axis=1)

test=test.drop('url',axis=1)

test=test.drop('text',axis=1)
train.shape

test.shape
X_train, X_test, y_train, y_test = train_test_split(train,ytrain, test_size=0.1)
clfr = RidgeClassifier()

clfr.fit(X_train, y_train)



y_pred = clfr.predict(X_test)

print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))


#sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

#sample_submission["target"] = clfr.predict(test)

#sample_submission.head()

#sample_submission.to_csv("submission.csv", index=False)

#set(train.columns).difference(set(test.columns))
set(train.columns).difference(set(test.columns))
#train.astype('float32')

train=train.iloc[1:]

ytrain=ytrain.iloc[1:]
from keras.models import Sequential

from keras.layers import Dense

import matplotlib.pyplot as plt



# fix random seed for reproducibility



# load pima indians dataset



# split into input (X) and output (Y) variables

ada=keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)

# create model

model = Sequential()

model.add(Dense(12, input_dim=19893, kernel_initializer='uniform', activation='relu'))

model.add(Dense(8, kernel_initializer='uniform', activation='relu'))

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer=ada, metrics=['accuracy'])

# Fit the model

history = model.fit(train,ytrain,validation_split=0.2, epochs=150,batch_size=10,verbose=2)

# list all data in history

print(history.history.keys())
[test_loss,test_acc]=model.evaluate(X_test,y_test)

print(test_loss,test_acc)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = model.predict(test)

sample_submission.head()

#sample_submission.to_csv("submission.csv")

#set(train.columns).difference(set(test.columns))
test.shape
train.shape

test