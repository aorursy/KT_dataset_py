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
os.chdir("../input")
df=pd.read_csv("train.csv")
df.head()
df["text"]=df["text"].apply(lambda x: x.lower())
import string

from nltk.corpus import stopwords

stopwords=stopwords.words("english")

punctuations=string.punctuation
def _clean(text):

#     print(text)

    text=[w for w in text if w not in punctuations]

    text="".join(text)

    text=text.split()

    text=[w for w in text if w not in stopwords ]

    

    text=" ".join(text)

    return text

_clean("I am a very bad girl")
df["text"]=df["text"].apply(lambda x: _clean(x))
from nltk.stem import  WordNetLemmatizer

from nltk import word_tokenize

lemma=WordNetLemmatizer()
def _lemma(text):

    words=word_tokenize(text)

    lemma_words=[lemma.lemmatize(word,"v") for word in words]

    return " ".join(lemma_words)
df["text"]=df["text"].apply(_lemma)
df.head()
from sklearn.feature_extraction.text import TfidfVectorizer

df=df.drop(columns="id", axis=1)
df.head()
tfidf=TfidfVectorizer()

tf=tfidf.fit_transform(df["text"])

df["text"][2]
df.shape
df.groupby(["author"]).count()
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(tf,df["author"],test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=0)
model.fit(train_x,train_y)
y_pred=model.predict(test_x)
from sklearn.metrics import f1_score, accuracy_score
f1_score(test_y,y_pred, average="micro")
from sklearn.feature_extraction.text  import CountVectorizer,TfidfVectorizer

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer
nb=Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',MultinomialNB())])
x,x_t,y,y_t=train_test_split(df["text"],df["author"],test_size=0.3,random_state=0)
nb.fit(x,y)
y_p=nb.predict(x_t)
f1_score(y_p,y_t,average="micro")
from sklearn.linear_model import SGDClassifier
sgd=Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('clf',SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None))])
sgd.fit(x,y)
y_pp=sgd.predict(x_t)
f1_score(y_pp,y_t,average="micro")