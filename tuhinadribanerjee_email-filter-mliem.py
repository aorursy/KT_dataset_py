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
df = pd.read_csv("../input/spam-not-spam/spam_or_not_spam (medit).csv",encoding = 'latin-1')
df
df.shape
df.drop_duplicates(inplace=True)

df.shape
positive_sentiment_count = len(df[df.label == 1])

negative_sentiment_count = len(df[df.label == 0])

print(positive_sentiment_count,negative_sentiment_count,sep="\n")
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)

plt.figure(figsize = (6,5))

ax = sns.countplot(x='label',data=df)
dp = float("NaN")

df.replace("", dp, inplace = True)

df.drop_duplicates(inplace = True)

df.dropna(inplace = True)

df.dropna(subset = ["email"],inplace = True)

df.shape

print(df.shape)

print(df.isnull().sum())

print(df)
from nltk.corpus import stopwords

stop_words = stopwords.words('english') #This part is a reference from maam's code
import string

def process_text(email):

    nopunc = [char for char in email if char in string.punctuation] 

    nopunc = ''.join(nopunc)

    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return clean_words
print(df['email'].head().apply(process_text))

df.shape
df
from sklearn.feature_extraction.text import CountVectorizer

testdata = CountVectorizer(analyzer=process_text).fit_transform(df['email'])
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(testdata, df['label'],test_size = 0.30,random_state=0)

print(testdata)

print(testdata.shape)
from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB().fit(X_train, y_train)

#print the predictions

print(classifier.predict(X_train))

print(y_train.values)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pred= classifier.predict(X_train)

print(classification_report(y_train,pred))

print()



print('Confusion matrix : \n', confusion_matrix (y_train,pred))

print()

print('Accuracy : ', accuracy_score(y_train, pred))
print(classifier.predict(X_test))
print(y_test.values)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pred= classifier.predict(X_test)

print(classification_report(y_test,pred))

print()
print('Confusion matrix : \n', confusion_matrix (y_test,pred))

print()

print('Accuracy : ', accuracy_score(y_test,pred))