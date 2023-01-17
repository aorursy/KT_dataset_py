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
data=pd.read_csv('../input/spam.csv',encoding='latin1')
data=data[['v1','v2']]
X=data.v2

y=data.v1
import seaborn as sns

sns.countplot(y)
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

CV=CountVectorizer()

X=CV.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
from sklearn.linear_model import LogisticRegression

LR=LogisticRegression()

LR.fit(X_train,y_train)

y_pred=LR.predict(X_test)
from sklearn.metrics import accuracy_score

print('Accuracy=',accuracy_score(y_pred,y_test))