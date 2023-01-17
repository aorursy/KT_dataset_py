import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('../input/kakr-4th-competition/train.csv', index_col='id')

print(df.shape)

df.head()
fig, axes = plt.subplots(2, 3, figsize=(11, 5), dpi=100)



kwargs = {

    'edgecolor': 'black', 

    'palette': sns.color_palette('Paired')}

i = 0

for col, values in df.select_dtypes(np.object).iteritems():

    if values.nunique() < 10:

        sns.countplot(

            y=values, 

            order=values.value_counts().index, 

            ax=axes.flat[i], **kwargs)

        

        axes.flat[i].grid(axis='x')

        axes.flat[i].set_ylabel('')

        axes.flat[i].set_title(f'{col}'.capitalize())

        i += 1

        

fig.tight_layout()

plt.show()
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=100)

i = 0

for col, values in df.select_dtypes(np.object).iteritems():

    if values.nunique() > 10 and col != 'native_country':

        sns.countplot(y=values, order=values.value_counts().index, ax=axes.flat[i], **kwargs)

        axes.flat[i].grid(axis='x')

        axes.flat[i].set_ylabel('')

        axes.flat[i].set_title(f'{col}'.capitalize())

        i += 1

        

fig.tight_layout()

plt.show()
fig, ax = plt.subplots(figsize=(5, 8), dpi=100)



sns.countplot(y='native_country', order=values.value_counts().index, data=df, ax=ax, **kwargs)

ax.grid(axis='x')

ax.set_ylabel('')

        

fig.tight_layout()

plt.show()
fig, axes = plt.subplots(3, 2, figsize=(11, 9), dpi=100)



i = 0

for col, values in df.select_dtypes(np.object).iteritems():

    if values.nunique() < 10 and col != 'sex':

        sns.countplot(

            y=values, hue='sex', data=df, order=values.value_counts().index, ax=axes.flat[i], **kwargs)

        axes.flat[i].grid(axis='x')

        axes.flat[i].set_ylabel('')

        axes.flat[i].set_title(f'{col}'.capitalize())

        axes.flat[i].legend(

            loc='lower right', frameon=True, framealpha=1,

            shadow=False, fancybox=False, edgecolor='black')

        i += 1



axes.flat[-1].set_visible(False)

fig.tight_layout()

plt.show()
fig, axes = plt.subplots(3, 2, figsize=(11, 9), dpi=100)



i = 0

for col, values in df.select_dtypes(np.object).iteritems():

    if values.nunique() < 10 and col != 'race':

        sns.countplot(

            y=values, hue='race', data=df, order=values.value_counts().index, ax=axes.flat[i], **kwargs)

        axes.flat[i].grid(axis='x')

        axes.flat[i].set_ylabel('')

        axes.flat[i].set_title(f'{col}'.capitalize())

        axes.flat[i].legend(

            loc='lower right', frameon=True, framealpha=1,

            shadow=False, fancybox=False, edgecolor='black')

        i += 1



axes.flat[-1].set_visible(False)

fig.tight_layout()

plt.show()
fig, axes = plt.subplots(1, 2, figsize=(11, 9), dpi=100)



i = 0

for col, values in df.select_dtypes(np.object).iteritems():

    if values.nunique() > 10 and col != 'native_country':

        sns.countplot(

            y=values, hue='sex', data=df, order=values.value_counts().index, ax=axes.flat[i], **kwargs)

        axes.flat[i].grid(axis='x')

        axes.flat[i].set_ylabel('')

        axes.flat[i].set_title(f'{col}'.capitalize())

        axes.flat[i].legend(

            loc='upper right', frameon=True, framealpha=1,

            shadow=False, fancybox=False, edgecolor='black')

        i += 1



fig.tight_layout()

plt.show()
# EDA는 나중에...
submission = pd.read_csv('../input/kakr-4th-competition/sample_submission.csv')

test = pd.read_csv('../input/kakr-4th-competition/test.csv', index_col='id')

train = pd.read_csv('../input/kakr-4th-competition/train.csv', index_col='id')
categories = {}

for col in train.select_dtypes(np.object).columns:

    categories[col] = {v: k for k, v in enumerate(train[col].value_counts().index.tolist())}



categories['native_country'].update({'Holand-Netherlands': -1})

    

for col in train.select_dtypes(np.object).columns:

    train[col] = train[col].replace(categories[col])

    if col != 'income':

        test[col] = test[col].replace(categories[col])
X = train.iloc[:, :-1].values

y = train.iloc[:, -1].values

test = test.values



# 럭키 7, max_depth=7, OOF는 나중에.... 0.85

rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42).fit(X, y) 

y_preds = rf.predict(test)

print(np.unique(y_preds, return_counts=True))



submission['prediction'] = y_preds

submission.to_csv('submission.csv', index=False)
params = {

    'objective': 'binary:logistic',

    'eta': 0.1, 

    'seed': 42,

    'max_depth': 7,

    'subsample': 0.8,

    'nthread': 4,

    'eval_metric': 'logloss'

}



train_preds = np.zeros(len(X))

test_preds = np.zeros(len(test))



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_index, valid_index) in enumerate(skf.split(X, y)):    

    dtrain = xgb.DMatrix(X[train_index], label=y[train_index])

    dvalid = xgb.DMatrix(X[valid_index], label=y[valid_index])



    bst = xgb.train(params, dtrain, num_boost_round=400, evals=[(dtrain, 'train'), (dvalid, 'valid')], verbose_eval=200)

    train_preds[valid_index] = bst.predict(dvalid)

    test_preds += bst.predict(xgb.DMatrix(test)) / skf.n_splits



test_preds = np.where(test_preds > 0.5, 1, 0)

print(test_preds)

print(np.unique(test_preds, return_counts=True))

submission['prediction'] = test_preds

submission.to_csv('submission.csv', index=False)