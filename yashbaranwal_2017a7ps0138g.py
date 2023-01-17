# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

np.random.seed()

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from pprint import pprint

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv')

test = pd.read_csv('/kaggle/input/bits-f464-l1/test.csv')
l = [i for i in train.columns if train[i].nunique() == 1]

l
train.drop(l, axis = 1, inplace = True)

test.drop(l, axis = 1, inplace = True)

train.columns[1:98]
cor = train[train.columns[1:98]].corr()
ans = abs(cor['b63'])#Selecting highly correlated features for the feature in the bracket (here, b63)

relevant_feat = ans[ans>0.9]

relevant_feat
unique = train.columns[1:98]

for i in unique:

    cor_target = abs(cor[i]) #Selecting highly correlated features

    relevant_features = cor_target[cor_target>0.9]  

    unwanted = list(relevant_features.index)

    unique = [x for x in unique if x not in unwanted]

    unique.append(i)

print(unique)

print('Number of features in unique = '+str(len(unique)))
all_feat = train.columns[1:98]

drop = ['b13', 'b39', 'b40', 'b46'] #These are features with extreme outliers

X = train[unique] #Excluding features that are highly correlated

y = train['label']



X.drop([i for i in drop if i in X.columns], axis=1, inplace = True) #Dropping features with extreme outliers

len(X.columns)
X_test = test[X.columns] #Selecting the same features from the test set.

X_test.shape
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor



ab_rgr = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100, random_state=42)

ab_rgr.fit(X,y)
y_pred = ab_rgr.predict(X_test)
pprint(y_pred[:20])
ids = test['id']

final = pd.DataFrame()

final['id'] = ids

final['label'] = y_pred

final.to_csv(r'final_best_submission.csv', index = False) #This file name is the .csv file that was submitted