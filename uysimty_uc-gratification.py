import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import roc_auc_score

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm



import os

print(os.listdir("../input"))
import warnings

warnings.filterwarnings('ignore')
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.shape
train_data.head()
test_data.head()
test_data.columns
fig, ax = plt.subplots(1, 3, figsize=(16,3), sharey=True)



train_data['stealthy-beige-pinscher-golden'].hist(bins=50, ax=ax[0])

train_data['dorky-peach-sheepdog-ordinal'].hist(bins=50, ax=ax[1])

train_data['snazzy-harlequin-chicken-distraction'].hist(bins=50, ax=ax[2])
for col in train_data.columns:

    unicos = train_data[col].unique().shape[0]

    if unicos < 1000:

        print(col, unicos)
train_data["wheezy-copper-turtle-magic"].hist()
oof = np.zeros(train_data.shape[0])

cols = [c for c in train_data.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]



for i in tqdm(range(512)):

    train_group = train_data[train_data['wheezy-copper-turtle-magic'] == i]

    train_X = train_group[cols]

    train_y = train_group['target']

    idx = train_X.index

    train_X.reset_index(drop=True,inplace=True)

    

    train_X = StandardScaler().fit_transform(PCA(n_components=40, random_state=4).fit_transform(train_X))

    

    skf = StratifiedKFold(n_splits=10, random_state=42)

    for train_index, test_index in skf.split(train_X, train_y):

        clf = QuadraticDiscriminantAnalysis()

        clf.fit(train_X[train_index], train_y.iloc[train_index])

        oof[idx[test_index]] = clf.predict_proba(train_X[test_index])[:,1]
print(roc_auc_score(train_data['target'], oof))
preds = np.zeros(len(test_data))



for i in tqdm(range(512)):

    train2 = train_data[train_data['wheezy-copper-turtle-magic']==i]

    test2 = test_data[test_data['wheezy-copper-turtle-magic']==i]

    train_idx = train2.index 

    test_idx = test2.index

    train2.reset_index(drop=True, inplace=True)

    

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = StandardScaler().fit_transform(PCA(n_components=40, random_state=4).fit_transform(data[cols]))

    train3 = data2[:train2.shape[0]]

    test3 = data2[train2.shape[0]:]

    

    clf = QuadraticDiscriminantAnalysis()

    clf.fit(train3, train2['target'])

    preds[test_idx] = clf.predict_proba(test3)[:,1]

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv', index=False)