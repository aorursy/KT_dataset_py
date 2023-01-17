# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
twitter=pd.read_csv('../input/train.csv',encoding='latin-1')
from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer()
X=twitter.SentimentText
y=twitter.Sentiment
X=vect.fit_transform(X)
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(X,y)
from sklearn.cross_validation import cross_val_score
cross=cross_val_score(nb,X,y,cv=10)
cross.mean()

# Any results you write to the current directory are saved as output.
