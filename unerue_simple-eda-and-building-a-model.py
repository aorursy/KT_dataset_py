# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

from scipy import stats



sns.set_style(style='white')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape, test.shape)
train.head()
train.info()
# Dependent variable = categorical

# Independent variables = numerical or 
fig, ax = plt.subplots(figsize=(5, 3), dpi=100)

train['target'].value_counts().plot(kind='bar', ax=ax)

for p in ax.patches:

    ax.annotate('{:,}'.format(p.get_height()), (p.get_x() + .13, p.get_height() + 3000),  fontsize=10)



ax.tick_params(axis='both', rotation=0, labelleft=False)

ax.set_ylim(0, 200000)

ax.grid(axis='y', linestyle='--')

ax.set_title('')

fig.tight_layout()

plt.show()
# Class imbalance

# Too many columns, so we will not get much information at once.

print(len(train.columns))
melted1 = pd.melt(train.iloc[:, np.r_[1, 2:42]], id_vars='target')

melted2 = pd.melt(train.iloc[:, np.r_[1, 42:82]], id_vars='target')

melted3 = pd.melt(train.iloc[:, np.r_[1, 82:122]], id_vars='target')

melted4 = pd.melt(train.iloc[:, np.r_[1, 122:162]], id_vars='target')

melted5 = pd.melt(train.iloc[:, np.r_[1, 162:202]], id_vars='target')
fig, axes = plt.subplots(5, 1, figsize=(25, 15), dpi=100)



for i, df in enumerate([melted1, melted2, melted3, melted4, melted5]):

    sns.boxplot(x='variable', y='value', hue='target', data=df, ax=axes.flat[i])

    axes.flat[i].grid(axis='x', linestyle='--')

    axes.flat[i].legend(loc='upper right', ncol=2, frameon=True)

fig.tight_layout()

plt.show()
# Statistical test

# Check p-values

# ANOVA
result = pd.DataFrame(columns=['var_names', 'p-values'])

result['var_names'] = train.columns[2:].tolist()

p_vals = []

for col in train.columns:

    if col not in ['ID_code', 'target']:

        _ = train.loc[:, ['target', col]].pivot(columns='target')

        statics, p_value = stats.f_oneway(_.iloc[:,0].dropna(), _.iloc[:,1].dropna())

        p_vals.append(p_value)

result['p-values'] = p_vals

result = result.assign(disparity=np.log(1./result['p-values'].values))

result.sort_values(by='disparity', ascending=False, inplace=True)



fig, ax = plt.subplots(figsize=(5, 35), dpi=100)

sns.barplot(y='var_names', x='disparity', data=result, color='lightsalmon', ax=ax)

plt.show()
print(len(train.columns))

print(len(result[result['p-values'] < 0.05]))
logit_mod = sm.Logit(train['target'], train.iloc[:, 2:])

logit_res = logit_mod.fit(disp=0)

print(logit_res.summary())
variables = logit_res.pvalues[logit_res.pvalues < 0.05].index.tolist()

print(len(variables))
import lightgbm as lgb

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from sklearn.model_selection import StratifiedKFold
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
train_preds = np.zeros(len(train))

test_preds = np.zeros(len(test))



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



for train_index, valid_index in skf.split(train[variables], train['target']):

    train_data = lgb.Dataset(train.loc[train_index, variables], 

                             label=train.loc[train_index, 'target'])

    valid_data = lgb.Dataset(train.loc[valid_index, variables], 

                             label=train.loc[valid_index, 'target'])

    

    bst = lgb.train(param, train_data, num_boost_round=2000, valid_sets=valid_data, 

                    verbose_eval=500, early_stopping_rounds=30)

    train_preds[valid_index] = bst.predict(train.loc[valid_index, variables], 

                                           num_iteration=bst.best_iteration)

    test_preds += bst.predict(test[variables], num_iteration=bst.best_iteration) / 5



print('Accuracy {}'.format(accuracy_score(train['target'], np.where(train_preds > 0.5, 1, 0))))

print('ROC AUC Score: {}'.format(roc_auc_score(train['target'], train_preds)))
import itertools



fig, ax = plt.subplots(figsize=(5,5), dpi=100)

classes = train.target.unique()

cm = confusion_matrix(train.target, np.where(train_preds > 0.5, 1, 0))

cs = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

fig.colorbar(cs)

tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)

plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

    ax.text(j, i, format(cm[i, j]),

            horizontalalignment='center',

            color='white' if cm[i, j] > thresh else 'black')

ax.set_ylabel('True label')

ax.set_xlabel('Predicted label')

fig.tight_layout()

plt.show()
submission = pd.DataFrame({'ID_code': test['ID_code'],

                           'target': test_preds})

submission.to_csv('submission.csv', index=False)