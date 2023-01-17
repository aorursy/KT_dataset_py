import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

from scipy import stats

from sklearn.model_selection import KFold

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

import gc



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('dark')
def resumetable(df):

    #print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary
train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

submission = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')
print("Train shape: ", train.shape)

print("Test shape: ", test.shape)

print("Submission shape: ", submission.shape)
train.head()
summary = resumetable(train)

summary
total = len(train)

plt.figure(figsize=(10,6))

flatui = ["#e74c3c", "#34495e"]



g = sns.countplot(x='target', data=train, palette=flatui)

g.set_title("TARGET DISTRIBUTION", fontsize = 20)

g.set_xlabel("Target Vaues", fontsize = 15)

g.set_ylabel("Count", fontsize = 15)





sizes=[] # Get highest values in y

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=14) 

g.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights

plt.show()
bin_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
import matplotlib.gridspec as gridspec

grid = gridspec.GridSpec(3, 2)

plt.figure(figsize=(16,20))



for n, col in enumerate(train[bin_cols]): 

    ax = plt.subplot(grid[n]) # feeding the figure of grid

    sns.countplot(x=col, data=train, hue='target', palette=flatui) 

    ax.set_ylabel('Count', fontsize=15) # y axis label

    ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label

    ax.set_xlabel(f'{col} values', fontsize=15) # x axis label

    sizes=[] # Get highest values in y

    for p in ax.patches: # loop to all objects

        height = p.get_height()

        sizes.append(height)

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(height/total*100),

                ha="center", fontsize=14) 

    ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights

    

plt.show()



plt.show()
def ploting_cat_fet(df, cols, vis_row=5, vis_col=2):

    

    grid = gridspec.GridSpec(vis_row,vis_col) # The grid of chart

    plt.figure(figsize=(17, 35)) # size of figure



    # loop to get column and the count of plots

    for n, col in enumerate(train[cols]): 

        tmp = pd.crosstab(train[col], train['target'], normalize='index') * 100

        tmp = tmp.reset_index()

        tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)



        ax = plt.subplot(grid[n]) # feeding the figure of grid

        sns.countplot(x=col, data=train, order=list(tmp[col].values) , palette='Set2') 

        ax.set_ylabel('Count', fontsize=15) # y axis label

        ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label

        ax.set_xlabel(f'{col} values', fontsize=15) # x axis label



        # twinX - to build a second yaxis

        gt = ax.twinx()

        gt = sns.pointplot(x=col, y='Yes', data=tmp,

                           order=list(tmp[col].values),

                           color='black', legend=False)

        gt.set_ylim(tmp['Yes'].min()-5,tmp['Yes'].max()*1.1)

        gt.set_ylabel("Target %True(1)", fontsize=16)

        sizes=[] # Get highest values in y

        for p in ax.patches: # loop to all objects

            height = p.get_height()

            sizes.append(height)

            ax.text(p.get_x()+p.get_width()/2.,

                    height + 3,

                    '{:1.2f}%'.format(height/total*100),

                    ha="center", fontsize=14) 

        ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights





    plt.subplots_adjust(hspace = 0.5, wspace=.3)

    plt.show()
nom_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']
ploting_cat_fet(train, nom_cols, vis_row=5, vis_col=2)
ord_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3']
ploting_cat_fet(train, ord_cols, vis_row=5, vis_col=2)
tmp = pd.crosstab(train['ord_4'], train['target'], normalize='index') * 100

tmp = tmp.reset_index()

tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)

plt.figure(figsize=(15,12))



plt.subplot(211)

ax = sns.countplot(x='ord_4', data=train, order=list(tmp['ord_4'].values), palette='Set2') 

ax.set_ylabel('Count', fontsize=17) # y axis label

ax.set_title('ord_4 Distribution with Target %ratio', fontsize=20) # title label

ax.set_xlabel('ord_4 values', fontsize=17) # x axis label

# twinX - to build a second yaxis

gt = ax.twinx()

gt = sns.pointplot(x='ord_4', y='Yes', data=tmp,

                   order=list(tmp['ord_4'].values),

                   color='black', legend=False)

gt.set_ylim(tmp['Yes'].min()-5,tmp['Yes'].max()*1.1)

gt.set_ylabel("Target %True(1)", fontsize=16)



gt = ax.twinx()

gt = sns.pointplot(x='ord_4', y='Yes', data=tmp,

                   order=list(tmp['ord_4'].values),

                   color='black', legend=False)

gt.set_ylim(tmp['Yes'].min()-5,tmp['Yes'].max()*1.1)

gt.set_ylabel("Target %True(1)", fontsize=16)





plt.show()
date_cols = ['day', 'month']
ploting_cat_fet(train, date_cols, vis_row=6, vis_col=2)
# dictionary to map the feature

bin_dict = {'T':1, 'F':0, 'Y':1, 'N':0}



# Maping the category values in our dict

train['bin_3'] = train['bin_3'].map(bin_dict)

train['bin_4'] = train['bin_4'].map(bin_dict)

test['bin_3'] = test['bin_3'].map(bin_dict)

test['bin_4'] = test['bin_4'].map(bin_dict)
# Concatenating train and test data

test['target'] = 'test'

df = pd.concat([train, test], axis=0, sort=False)



print("Data shape:", df.shape)
print(f'Shape before dummy transformation: {df.shape}')

df = pd.get_dummies(df, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],\

                          prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], drop_first=True)

print(f'Shape after dummy transformation: {df.shape}')
# Seperating dataset into train and test set

train, test = df[df['target'] != 'test'], df[df['target'] == 'test'].drop('target', axis=1)

train['target'] = train['target'].astype(int)

del df
# Importing categorical options of pandas

from pandas.api.types import CategoricalDtype 



# seting the orders of our ordinal features

ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 

                                     'Master', 'Grandmaster'], ordered=True)

ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',

                                     'Boiling Hot', 'Lava Hot'], ordered=True)

ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',

                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)

ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',

                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',

                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)
# Transforming ordinal Features

train.ord_1 = train.ord_1.astype(ord_1)

train.ord_2 = train.ord_2.astype(ord_2)

train.ord_3 = train.ord_3.astype(ord_3)

train.ord_4 = train.ord_4.astype(ord_4)



# test dataset

test.ord_1 = test.ord_1.astype(ord_1)

test.ord_2 = test.ord_2.astype(ord_2)

test.ord_3 = test.ord_3.astype(ord_3)

test.ord_4 = test.ord_4.astype(ord_4)
# Geting the codes of ordinal categoy's - train

train.ord_1 = train.ord_1.cat.codes

train.ord_2 = train.ord_2.cat.codes

train.ord_3 = train.ord_3.cat.codes

train.ord_4 = train.ord_4.cat.codes



# Geting the codes of ordinal categoy's - test

test.ord_1 = test.ord_1.cat.codes

test.ord_2 = test.ord_2.cat.codes

test.ord_3 = test.ord_3.cat.codes

test.ord_4 = test.ord_4.cat.codes
train[['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4']].head()
# Transfer the cyclical features into two dimensional sin-cos features

# https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning



def date_cyc_enc(df, col, max_vals):

    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

    return df



train = date_cyc_enc(train, 'day', 7)

test = date_cyc_enc(test, 'day', 7) 



train = date_cyc_enc(train, 'month', 12)

test = date_cyc_enc(test, 'month', 12)



# NOTE, I discovered it on: kaggle.com/gogo827jz/catboost-baseline-with-feature-importance
### Credit of this features to: 

## https://www.kaggle.com/gogo827jz/catboost-baseline-with-feature-importance



import string



# Then encode 'ord_5' using ACSII values



# Option 1: Add up the indices of two letters in string.ascii_letters

train['ord_5_oe_add'] = train['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))

test['ord_5_oe_add'] = test['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))



# Option 2: Join the indices of two letters in string.ascii_letters

train['ord_5_oe_join'] = train['ord_5'].apply(lambda x:float(''.join(str(string.ascii_letters.find(letter)+1) for letter in x)))

test['ord_5_oe_join'] = test['ord_5'].apply(lambda x:float(''.join(str(string.ascii_letters.find(letter)+1) for letter in x)))



# Option 3: Split 'ord_5' into two new columns using the indices of two letters in string.ascii_letters, separately

train['ord_5_oe1'] = train['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))

test['ord_5_oe1'] = test['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))



train['ord_5_oe2'] = train['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))

test['ord_5_oe2'] = test['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))



for col in ['ord_5_oe1', 'ord_5_oe2', 'ord_5_oe_add', 'ord_5_oe_join']:

    train[col]= train[col].astype('float64')

    test[col]= test[col].astype('float64')
high_card = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_5']
from category_encoders.target_encoder import TargetEncoder

te = TargetEncoder()
train[high_card] = te.fit_transform(X = train[high_card], y = train['target'])

test[high_card] = te.transform(X = test[high_card])
train.head()
test.head()
X = train.drop(['id','target'], axis = 1)

y = train['target']

X_test = test.drop('id', axis = 1)
print("X:", X.shape)

print("y:", y.shape)

print("X_test:", X_test.shape)
params = {

    'bagging_fraction': 0.52,

    'boosting': 'goss',

    'feature_fraction': 0.5,

    'lambda_l1': 3.135897735211238,

    'lambda_l2': 1.8097983046367754,

    'learning_rate': 0.024895196236388753,

    'max_bin': 64,

    'max_depth': 2,

    'metric': 'auc',

    'min_data_in_bin': 176,

    'min_data_in_leaf': 144,

    'min_gain_to_split': 4.97,

    'num_leaves': 1393,

    'objective': 'binary',

    'other_rate': 0.4399622643268988,

    'top_rate': 0.1919072599467846,

    'is_unbalance': True,

    'random_state': 42

}
%%time



NFOLDS = 5

folds = KFold(n_splits=NFOLDS)



columns = X.columns

splits = folds.split(X, y)

y_preds = np.zeros(X_test.shape[0])

y_oof = np.zeros(X.shape[0])

score = 0



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns

  

for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    dtrain = lgb.Dataset(X_train, label=y_train)

    dvalid = lgb.Dataset(X_valid, label=y_valid)



    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)

    

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(X_valid)

    y_oof[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    y_preds += clf.predict(X_test) / NFOLDS

    

    del X_train, X_valid, y_train, y_valid

    gc.collect()

    

print(f"\nMean AUC = {score}")

print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")
feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)

feature_importances.to_csv('feature_importances.csv')



plt.figure(figsize=(16, 16))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));
submission = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')
submission['target'] = y_preds
submission.head()
submission.to_csv('submission.csv', index=False)