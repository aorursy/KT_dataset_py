import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")
train = pd.read_csv('../input/bnp-paribas-cardif-claims-management/train.csv.zip')

train.shape
test = pd.read_csv('../input/bnp-paribas-cardif-claims-management/test.csv.zip')

test.shape
train.head(20)
train.describe()
test.head()
train.describe()
train.describe(include="O")
missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar(figsize=(15,6))

plt.show()
train_val = train.copy()

test_val = test.copy()



# fill NaN values with -99

for f in train_val.columns:

    if train_val[f].dtype == 'float64':

        train_val[f].fillna(-99, inplace=True)

        test_val[f].fillna(-99, inplace=True)

        #train_val.loc[:,f][np.isnan(train_val[f])] = -99

        #test_val[f][np.isnan(test_val[f])] = -99

        

    elif train_val[f].dtype == 'object':

        train_val[f][train_val[f] != train[f]] = -99

        test_val[f][test_val[f] != test[f]] = -99
train_val.head()
from sklearn import preprocessing



for f in train_val.columns:

    if train_val[f].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(np.unique(list(train_val[f].values)  + list(test_val[f].values)))

        train_val[f]   = lbl.transform(list(train_val[f].values))

        test_val[f]  = lbl.transform(list(test_val[f].values))



#train_val.head()
train_X = train_val.drop('target', axis=1)

train_y = train_val.target
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=100, random_state=1, oob_score=True)

rfc = rfc.fit(train_X, train_y)

print("%.4f" % rfc.oob_score_)
pred = rfc.predict(test_val)

pred_prob = rfc.predict_proba(test_val)
importances = rfc.feature_importances_

std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")

for f in range(10):

    print("%d. %s (%f)" % (f + 1, train_X.columns[indices[f]], importances[indices[f]]))
Ns = 20

plt.title('Feature Importance')

plt.bar(range(Ns),importances[indices[0:Ns]], yerr=std[indices[0:Ns]], color='r', align='center')

plt.xticks(range(Ns), train_X.columns[indices[0:Ns]], rotation=90)

plt.xlim([-1, Ns])

plt.tight_layout()

plt.show()
sample_sub = pd.read_csv('../input/bnp-paribas-cardif-claims-management/sample_submission.csv.zip')

sample_sub.head(10)
pred_prob_df = pd.DataFrame(pred_prob)

submit = pd.concat([test['ID'],pred_prob_df], axis=1)

submit.head()
submit.rename(columns={1: 'PredictedProb'}, inplace=True)

submit.to_csv("submit_rfc02.csv", columns=['ID','PredictedProb'], index=None)