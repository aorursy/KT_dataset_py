

import numpy as np 

import pandas as pd 

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt





import os

print(os.listdir("../input"))

data = pd.read_csv('../input/data.csv')

data.head()
data.describe()
data.isnull().sum()
data.dropna(axis=1, inplace=True)

data['diagnosis'] = data.diagnosis.apply(lambda x: 1 if x == 'M' else 0)

labels = data.diagnosis

data.drop(['id', 'diagnosis'], axis=1, inplace=True)
data.head()
from scipy.stats import chi2_contingency

from scipy.stats import chi2

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2 as sklearn_chi2
bestfeatures = SelectKBest(score_func=sklearn_chi2, k=5)

x_new = bestfeatures.fit_transform(data,labels)

print(bestfeatures.scores_)
x_new = bestfeatures.fit_transform(data,labels)

x_new