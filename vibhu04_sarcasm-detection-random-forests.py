# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_json('/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json',lines=True)
df.head()
print('Sarcastic: ',len(df[df['is_sarcastic'] == 1]))

print('Not Sarcastic: ',len(df[df['is_sarcastic'] == 1]))
del df['article_link']
df.head()
x = df['headline'].values

y = df['is_sarcastic'].values 
#removing punct

import string

punct = string.punctuation
cleaned_x = []

for word in x:

    sent = ''

    for char in word:

        if char not in punct:

            char = char.lower()

            sent += char

    cleaned_x.append(sent)
#later try removing stop words and check accuracy

import nltk

from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

cleaned_X = []

for sent in cleaned_x:

    sent_cleaned = ''

    for word in sent.split():

        if word not in stops:

            sent_cleaned += word

            sent_cleaned += ' '

    cleaned_X.append(sent_cleaned)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1000)

vectorizer.fit(cleaned_x)
x_data = vectorizer.transform(cleaned_x)
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x_data,y,random_state=1)

x_train.shape,y_train.shape,x_test.shape,y_test.shape
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=500,random_state=42)

rfc.fit(x_train,y_train)
preds = rfc.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print('accuracy: ', accuracy_score(y_test,preds))

print(confusion_matrix(y_test,preds))
count_vec = CountVectorizer(max_features = 1000, ngram_range=(1,2))

count_vec.fit(cleaned_x)
x_feat_ngrams = count_vec.transform(cleaned_x)
x_feat_ngrams = np.array(x_feat_ngrams.todense())
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x_feat_ngrams,y,random_state=1)

x_train.shape,y_train.shape,x_test.shape,y_test.shape
rfc = RandomForestClassifier(n_estimators=500,random_state=42)

rfc.fit(x_train,y_train)
preds = rfc.predict(x_test)
print('accuracy: ', accuracy_score(y_test,preds))

print(confusion_matrix(y_test,preds))
from sklearn.feature_extraction.text import TfidfVectorizer
tfIdfVec = TfidfVectorizer(max_features=1000)

tfIdfVec.fit(cleaned_x)
tf_idf_x = tfIdfVec.transform(cleaned_x)
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(tf_idf_x.todense(),y,random_state=1)

x_train.shape,y_train.shape,x_test.shape,y_test.shape
rfc = RandomForestClassifier(n_estimators=500,random_state=42)

rfc.fit(x_train,y_train)
preds = rfc.predict(x_test)
print('accuracy: ', accuracy_score(y_test,preds))

print(classification_report(y_test,preds))