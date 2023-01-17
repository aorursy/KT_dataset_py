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
df = pd.read_csv("../input/train.csv")
df.head()
y = df['target']

id = df['ID_code']

df_t = df.drop(labels=['ID_code', 'target'], axis=1)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
x = df_t.values
x.shape
import lightgbm as lgb

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from sklearn.model_selection import StratifiedKFold
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42)
param = {

    'num_leaves': 5,

    'max_depth': 15,

    'save_binary': True,

    'seed': 42,

    'objective': 'binary',

    'boosting_type': 'gbdt',

    'verbose': 1,

    'metric': 'auc',

    'is_unbalance': True,

}
train_preds = np.zeros(len(df_t))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



for train_index, valid_index in skf.split(x, y):

    train_data = lgb.Dataset(df_t.loc[train_index], 

                             label=df.loc[train_index, 'target'])

    valid_data = lgb.Dataset(df_t.loc[valid_index], 

                             label=df.loc[valid_index, 'target'])

    

    bst = lgb.train(param, train_data, num_boost_round=2000, valid_sets=valid_data, 

                    verbose_eval=500, early_stopping_rounds=30)

    train_preds[valid_index] = bst.predict(df_t.loc[valid_index], 

                                           num_iteration=bst.best_iteration)

    #test_preds += bst.predict(test[variables], num_iteration=bst.best_iteration) / 5



print('Accuracy {}'.format(accuracy_score(df['target'], np.where(train_preds > 0.5, 1, 0))))

print('ROC AUC Score: {}'.format(roc_auc_score(df['target'], train_preds)))
preds = bst.predict(x_train)

print(accuracy_score(y_train,np.where(preds > 0.5, 1, 0)))
df_test = pd.read_csv('../input/test.csv')
id_test = df_test['ID_code']
df_test.drop(labels=['ID_code'], axis=1,inplace=True)
x_t = df_test.values
predictions = bst.predict(x_t)

predictions = np.where(predictions > 0.5, 0, 1)
data = {'ID_code': id_test, 'target': predictions}

out = pd.DataFrame(data)

out.to_csv('submission.csv', index=False)