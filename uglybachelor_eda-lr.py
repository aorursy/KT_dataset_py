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
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv', index_col='customerID')
data.head()
train = data.drop('Churn', axis=1)
label = data.Churn
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
col_to_enc = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
              'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
for col in col_to_enc:
    train[col+'_le'] = le.fit_transform(train[col])
train.drop(col_to_enc, axis=1, inplace=True)
import re
train['TotalCharges'].fillna(-1, inplace=True)
train['TotalCharges'] = train['TotalCharges'].apply(lambda x: re.findall(r"[\d]{1,4}\.?[\d]{1,2}", x))
def foo(x):
    if len(x) == 1:
        return x[0]
    else:
        return -1
train['TotalCharges'] = train['TotalCharges'].apply(foo)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, label, random_state=1)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
score
