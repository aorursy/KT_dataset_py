# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

data.head()
model=LogisticRegression()
from sklearn.model_selection import train_test_split

label=data['target']

data=data.drop('target',axis=1)

train_data,test_data,train_label,test_label=train_test_split(data,label,test_size=0.3)
model.fit(train_data,train_label)
pred=model.predict(test_data)
from sklearn.metrics import accuracy_score

print('Accuracy : ',accuracy_score(test_label,pred))
model2=DecisionTreeClassifier()

model2.fit(train_data,train_label)
pred2=model2.predict(test_data)
print('Accuracy : ',accuracy_score(test_label,pred2))