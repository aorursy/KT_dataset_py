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
train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train.head()
train_x=train['text']

train_y=train['target']
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X=vectorizer.fit_transform(train_x)
X.shape
X_test=vectorizer.transform(test['text'])
test.head()
X_test.shape
from sklearn import svm

cls=svm.SVC()

cls.fit(X,train_y)
y=cls.predict(X_test)
submission['target']=y
submission.to_csv('submissionsvm.csv',index=False)
from sklearn.neighbors import NearestCentroid

clf = NearestCentroid()

clf.fit(X,train_y)
y=clf.predict(X_test)
y
submission['target']=y

submission.to_csv('submissionneighbor.csv',index=False)
from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=2)

clf.fit(X,train_y)
y=clf.predict(X_test)

y
submission['target']=y

submission.to_csv('submissionrandomfor.csv',index=False)