# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import json
# Any results you write to the current directory are saved as output.
# learn from shivam bansal's kernel https://www.kaggle.com/shivamb/what-s-cooking-tf-idf-with-ovr-svm
train=json.load(open('../input/train.json'))
test=json.load(open('../input/test.json'))
train_data=[' '.join(doc['ingredients']) for doc in train]
test_data=[' '.join(doc['ingredients']) for doc in test]
tfidf=TfidfVectorizer(binary=True)
X=tfidf.fit_transform(train_data)
X_test=tfidf.transform(test_data)
X=X.astype('float16')
X_test=X_test.astype('float16')
target=[doc['cuisine'] for doc in train]
lb=LabelEncoder()
y=lb.fit_transform(target)
classifier=SVC(C=100,kernel='rbf',degree=3,gamma=1,coef0=1,shrinking=True,tol=0.001,probability=False,cache_size=200,class_weight=None,verbose=False,max_iter=-1,decision_function_shape=None,random_state=None)
model=OneVsRestClassifier(classifier,n_jobs=4)
model.fit(X[:5000],y[:5000])
y_test=model.predict(X_test)
y_pred=lb.inverse_transform(y_test)
