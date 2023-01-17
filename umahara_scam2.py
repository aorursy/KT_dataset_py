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
from collections import Counter

from sklearn.datasets import make_classification

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE
import pandas as pd

import numpy as np

df_train = pd.read_csv("/kaggle/input/1056lab-fraud-detection-in-credit-card/train.csv",index_col=0)

df_test = pd.read_csv("/kaggle/input/1056lab-fraud-detection-in-credit-card/test.csv",index_col=0)

#データ数とカラムの数

print('Train Shape : {}'.format(df_train.shape))

print('Test Shape : {}'.format(df_test.shape))
import matplotlib.pyplot as plt

import seaborn as sns

target_col = "Class"

sns.set()



sns.countplot(df_train[target_col])

plt.show()
cols = df_test.columns



for col in cols:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5)) # 1行2列で表示

    sns.distplot(df_train[col], ax=ax1)

    sns.distplot(df_test[col], ax=ax2)

    plt.show()
X = df_train.drop('Class',axis=1)

y = df_train['Class'].values
sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(X, y)
from sklearn.model_selection import GridSearchCV

clf = DecisionTreeClassifier()

params = {'criterion':('gini', 'entropy'), 'max_depth':[1, 2, 3, 4]}

gscv = GridSearchCV(clf, params, cv=5)

gscv.fit(X_res, y_res)
print('%.3f  %r' % (gscv.best_score_, gscv.best_params_))
from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier(max_depth=2)

clf.fit(X_res, y_res)
X_test = df_test.values

predict = clf.predict(X_test)
submit = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/sampleSubmission.csv',index_col=0)

submit['Class'] = predict

submit.to_csv('submission.csv', index=True)