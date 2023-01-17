import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.metrics import average_precision_score

import lightgbm as lgb
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
X = pd.concat([

    train.drop(columns=['ID_code', 'target']),

    test.drop(columns=['ID_code'])

], axis=0).values

y = np.append(

    np.zeros(len(train)),

    np.ones(len(test))

)
%%time

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

lgb_params = {

    'objective': 'binary',

    'metric': 'auc',

}

oof = np.zeros(len(y))



for trn_idx, val_idx in folds.split(X, y):

    X_trn = X[trn_idx]

    X_val = X[val_idx]

    y_trn = y[trn_idx]



    train_set = lgb.Dataset(X_trn, label=y_trn)

    clf = lgb.train(lgb_params, train_set)

    oof[val_idx] = clf.predict(X_val)

    

print('MAP:', average_precision_score(y, oof))

print('AUC:', roc_auc_score(y, oof))
sns.distplot(oof[y==0], bins=100)

sns.distplot(oof[y==1], bins=100)

plt.show()
oof_df = pd.DataFrame(data={0:oof[y==0], 1:oof[y==1]})

oof_df.describe(percentiles=np.linspace(0.1, 0.9, 9))