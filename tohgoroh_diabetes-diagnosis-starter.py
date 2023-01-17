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
df_train = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/test.csv', index_col=0)
df_train
df_test
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.countplot(x='Diabetes', data=df_train)

plt.show()
df_train_dummies = pd.get_dummies(df_train, columns=['Gender'], drop_first=True)

df_train_dummies
df_test_dummies = pd.get_dummies(df_test, columns=['Gender'], drop_first=True)

df_test_dummies
import matplotlib.pyplot as plt

%matplotlib inline

sns.pairplot(df_train_dummies, hue='Diabetes')

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier



X_train_dummies = df_train_dummies.drop(columns='Diabetes').values  # Numpy行列を取り出す

y_train_dummies = df_train_dummies['Diabetes'].values  # Numpy行列を取り出す



X_train, X_valid, y_train, y_valid = train_test_split(X_train_dummies, y_train_dummies, test_size=0.2, random_state=0)  # 訓練用と検証用に分ける



dtc = DecisionTreeClassifier()  # モデル

dtc.fit(X_train, y_train)  # 学習
!apt install graphviz

!pip install dtreeviz
from dtreeviz.trees import *

import graphviz



viz = dtreeviz(dtc, X_train, y_train, target_name="Diabetes", feature_names=df_train_dummies.columns, class_names=['Not diabetes', 'Diabetes'])

viz
from sklearn.metrics import roc_curve, auc



y_pred = dtc.predict_proba(X_valid)[:, 1]  # 予測

fpr, tpr, thresholds = roc_curve(y_valid, y_pred)  # ROC曲線を求める

auc(fpr, tpr)  # 評価
from sklearn.tree import DecisionTreeClassifier



dtc = DecisionTreeClassifier()

dtc.fit(X_train_dummies, y_train_dummies)  # 学習
X_test = df_test_dummies.values  # Numpy行列を取り出す

y_pred = dtc.predict_proba(X_test)[:, 1]  # 予測
submit = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/sampleSubmission.csv')

submit['Diabetes'] = y_pred

submit.to_csv('submission.csv', index=False)