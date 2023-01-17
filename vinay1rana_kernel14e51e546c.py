# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import CountVectorizer



from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

data.head()
def preprocessing(x):

    

    corpus = []

    for i in range(0,x.size):

        tweets = re.sub('[^a-zA-Z]',' ',x[i])

        tweets = tweets.lower()

        tweets = tweets.split()

        ps = PorterStemmer()

        tweets = [ps.stem(word) for word in tweets if not word in set(stopwords.words('english'))]

        tweets = ' '.join(tweets)

        corpus.append(tweets)

    return corpus



x = data['text'].values

y = data['target'].values

x = preprocessing(x)


cv = CountVectorizer()

tweets= cv.fit_transform(x).toarray()



count = 0

#Function to check the zeros frequency values in our data

def checkZeros():

    for n in range(0,x.size):

        if(tweets[n].sum() != 0):

            count += 1
x_train,x_test,y_train,y_test = train_test_split(tweets,y,test_size=0.2)
xgb = XGBClassifier()

xgb_train = xgb.fit(x_train,y_train)

xgb_predict = xgb_train.predict(x_test)

f1_score(y_test,xgb_predict,average='binary')*100
#get sample file for creating submission file

test_file = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")



test_data = []

testing = test_file['text'].values

test_data = preprocessing(testing)

test_data = cv.transform(test_data).toarray()



sample_submission["target"] = xgb_train.predict(test_data)

sample_submission.to_csv("submission.csv", index=False)

sample_submission.tail()