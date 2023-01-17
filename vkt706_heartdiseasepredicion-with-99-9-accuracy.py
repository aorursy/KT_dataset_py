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
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.describe()
import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

data.hist(figsize=(20,20))
x = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',

       'exang', 'oldpeak', 'slope', 'ca', 'thal']]
from sklearn.preprocessing import StandardScaler as ss

x = ss().fit_transform(x.values)

y = data['target'].values
from sklearn.model_selection import train_test_split

X_train, x_test, Y_train,y_test = train_test_split(x, y, shuffle = True, random_state= 42, stratify = y)
from sklearn.metrics import accuracy_score

def acc(model):

    y_pred = model.predict(x_test)

    print(accuracy_score(y_pred, y_test))
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(penalty='l2')

lr.fit(X_train, Y_train)
y_pred = lr.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_pred, y_test)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



import xgboost as xg

from xgboost import XGBClassifier



model = xg.XGBClassifier()

model.fit(x,y)

probs = model.predict_proba(x_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# calculate AUC

auc = roc_auc_score(y_test, probs)

print('AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, probs)

# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

plt.plot(fpr, tpr, marker='.')

plt.show()