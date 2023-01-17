import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.linear_model import Ridge, LogisticRegression

import time

from sklearn import preprocessing

import warnings

import datetime

warnings.filterwarnings("ignore")

import gc

from tqdm import tqdm

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from scipy.stats import describe, rankdata

%matplotlib inline



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import mean_squared_error, roc_auc_score

import xgboost as xgb



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Loading Train and Test Data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("{} observations and {} features in train set.".format(train.shape[0],train.shape[1]))

print("{} observations and {} features in test set.".format(test.shape[0],test.shape[1]))
train.head()
test.head()
train.target.describe()
train[train.columns[2:]].describe()
test[test.columns[1:]].describe()
plt.figure(figsize=(12, 5))

plt.hist(train['0'].values, bins=20)

plt.title('Histogram 0 train counts')

plt.xlabel('Count')

plt.ylabel('Target')

plt.show()
plt.figure(figsize=(12, 5))

plt.hist(train['1'].values, bins=20)

plt.title('Histogram 1 train counts')

plt.xlabel('Count')

plt.ylabel('Target')

plt.show()
plt.figure(figsize=(12, 5))

plt.hist(train['123'].values, bins=20)

plt.title('Histogram 123 counts')

plt.xlabel('Count')

plt.ylabel('Target')

plt.show()
plt.figure(figsize=(12, 5))

plt.hist(test['0'].values, bins=20)

plt.title('Histogram 0 test counts')

plt.xlabel('Count')

plt.ylabel('Target')

plt.show()
plt.figure(figsize=(12, 5))

plt.hist(test['1'].values, bins=20)

plt.title('Histogram 1 test counts')

plt.xlabel('Count')

plt.ylabel('Target')

plt.show()
plt.figure(figsize=(12, 5))

plt.hist(test['123'].values, bins=20)

plt.title('Histogram 123 test counts')

plt.xlabel('Count')

plt.ylabel('Target')

plt.show()
train_test = pd.concat([train[train.columns[2:]], test[test.columns[1:]]])
train_test.shape
corr = train_test.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
AUCs = []

Ginis = []





for i in range(300):

    AUC = roc_auc_score(train.target.values, train[str(i)].values)

    AUCs.append(AUC)

    Gini = 2*AUC - 1

    Ginis.append(Gini)

np.sort(np.abs(Ginis))[::-1]
np.argsort(np.abs(Ginis))[::-1]
[Ginis[k] for k in np.argsort(np.abs(Ginis))[::-1][:13]]
roc_auc_score(train.target.values,train['33'].values)
roc_auc_score(train.target.values,train['65'].values)
roc_auc_score(train.target.values,-train['217'].values)
roc_auc_score(train.target.values,-train['117'].values)
roc_auc_score(train.target.values,-train['91'].values)
roc_auc_score(train.target.values,-train['295'].values)
roc_auc_score(train.target.values,train['24'].values)
roc_auc_score(train.target.values,train['199'].values)
roc_auc_score(train.target.values,-train['80'].values)
roc_auc_score(train.target.values,-train['73'].values)
roc_auc_score(train.target.values,-train['194'].values)
roc_auc_score(train.target.values,0.146*train['33'].values + 0.12*train['65'].values-0.06*train['217'].values-0.05*train['117'].values

             -0.05*train['91'].values-0.05*train['295'].values+0.05*train['24'].values+0.05*train['199'].values-

             0.05*train['80'].values- 0.05*train['73'].values-0.05*train['194'].values)
preds = (0.146*test['33'].values + 0.12*test['65'].values-0.06*test['217'].values-0.05*test['117'].values

             -0.05*test['91'].values-0.05*test['295'].values+0.05*test['24'].values+0.05*test['199'].values-

             0.05*test['80'].values- 0.05*test['73'].values-0.05*test['194'].values)

preds = rankdata(preds)/preds.shape[0]

preds
sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission['target'] = preds

sample_submission.to_csv('submission.csv', index=False)
pca = PCA(n_components=0.99)

pca.fit(train_test.values)
pca.n_components_
pca = PCA(n_components=0.9)

pca.fit(train_test.values)

pca.n_components_
Sum_of_squared_distances = []

K = range(1,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(train_test)

    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
NN = 80



train_pred = 0



gini_list = [Ginis[k] for k in np.argsort(np.abs(Ginis))[::-1][:NN]]



for i in range(NN):

    if gini_list[i] > 0:

        train_pred += train[str(np.argsort(np.abs(Ginis))[::-1][i])].values

    else:

        train_pred -= train[str(np.argsort(np.abs(Ginis))[::-1][i])].values

        

roc_auc_score(train.target.values, train_pred)
test_pred = 0



gini_list = [Ginis[k] for k in np.argsort(np.abs(Ginis))[::-1][:NN]]



for i in range(NN):

    if gini_list[i] > 0:

        test_pred += test[str(np.argsort(np.abs(Ginis))[::-1][i])].values

    else:

        test_pred -= test[str(np.argsort(np.abs(Ginis))[::-1][i])].values
test_pred = rankdata(test_pred)/test_pred.shape[0]

test_pred
sample_submission['target'] = test_pred

sample_submission.to_csv('submission_80.csv', index=False)