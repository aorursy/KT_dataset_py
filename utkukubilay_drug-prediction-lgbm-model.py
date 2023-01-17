import multiprocessing

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import gc

from time import time

import pandas as pd

import numpy as np

import datetime

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import log_loss

import os

import time



sns.set()

import random

from pylab import rcParams



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        warnings.simplefilter('ignore')
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')



train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")



train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
test_features.shape, submission.shape, train_targets_scored.shape, train_features.shape, train_targets_nonscored.shape
for feature in ['cp_type', 'cp_dose']:

    trans = LabelEncoder()

    trans.fit(list(train_features[feature].astype(str).values) + list(test_features[feature].astype(str).values))

    train_features[feature] = trans.transform(list(train_features[feature].astype(str).values))

    test_features[feature] = trans.transform(list(test_features[feature].astype(str).values))
def get_redundant_pairs(df):

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Absolute Correlations !")

print(get_top_abs_correlations(train_features.select_dtypes(include=["float64"]), 30))

drop_list_for_cor = ["c-13", "c-38","c-4","c-42","c-2"]

train_features.drop(drop_list_for_cor, axis = 1, inplace = True)

test_features.drop(drop_list_for_cor, axis = 1, inplace = True)


train_columns = train_features.columns.to_list()



g_list = [i for i in train_columns if i.startswith('g-')]

c_list = [i for i in train_columns if i.startswith('c-')]

columns = g_list + c_list



start = time.time()

cols = columns

above_90 = []



for i in range(0, len(cols)):

    for j in range(i+1, len(cols)):

        if abs(train_features[cols[i]].corr(train_features[cols[j]])) > 0.90:

            above_90.append(cols[i])

            



print(time.time()-start)

len(above_90)
liste = random.sample(above_90, 20)

train_columns = train_features.columns.to_list()



g_list = [i for i in train_columns if i.startswith('g-')]

c_list = [i for i in train_columns if i.startswith('c-')]

columns = g_list + c_list



start = time.time()

under_30 = []



for i in range(0, len(cols)):

    for j in range(i+1, len(cols)):

        if (abs(train_features[cols[i]].corr(train_features[cols[j]])) < 0.30):

            under_30.append(cols[i])

            

print(time.time()-start)
korelasyon_sikligi = pd.DataFrame([[x,under_30.count(x)] for x in set(under_30)])

korelasyon_sikligi.columns = ["columns_name","value"]

dusuk_cor = korelasyon_sikligi.sort_values(by="value")[-40:].columns_name.tolist()
len(dusuk_cor)
liste.extend(dusuk_cor)
len(liste)
%%time



numerik_cols = liste



kategorik=["cp_type","cp_time","cp_dose"]





for col in numerik_cols:

    for feat in kategorik:

        train_features[f'{col}_mean_group_{feat}']=train_features[col]/train_features.groupby(feat)[col].transform('mean')

        train_features[f'{col}_max_group_{feat}']=train_features[col]/train_features.groupby(feat)[col].transform('max')

        train_features[f'{col}_min_group_{feat}']=train_features[col]/train_features.groupby(feat)[col].transform('min')

        train_features[f'{col}_skew_group_{feat}']=train_features[col]/train_features.groupby(feat)[col].transform('skew')

        train_features[f'{col}_skew_group_{feat}']=train_features[col]/train_features.groupby(feat)[col].transform('std')

        
%%time



for col in numerik_cols:

    for feat in kategorik:

        test_features[f'{col}_mean_group_{feat}']=test_features[col]/test_features.groupby(feat)[col].transform('mean')

        test_features[f'{col}_max_group_{feat}']=test_features[col]/test_features.groupby(feat)[col].transform('max')

        test_features[f'{col}_min_group_{feat}']=test_features[col]/test_features.groupby(feat)[col].transform('min')

        test_features[f'{col}_skew_group_{feat}']=test_features[col]/test_features.groupby(feat)[col].transform('skew')

        test_features[f'{col}_skew_group_{feat}']=test_features[col]/test_features.groupby(feat)[col].transform('std')

        
features = [x for x in train_features.columns if x != 'sig_id']

print(len(features))
targets = [x for x in train_targets_scored.columns if x != 'sig_id']

print(f'Total Labels available : {len(targets)}')
X=train_features[features]

total_loss = 0
def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    

seed_everything(seed=42)



params = {'num_leaves': 490,

          'min_child_weight': 0.03,

          'feature_fraction': 0.55,

          'bagging_fraction': 0.9,

          'min_data_in_leaf': 150,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.01,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'binary_logloss',

          "verbosity": 0,

          'reg_alpha': 0.4,

          'reg_lambda': 0.6,

          'random_state': 47

         }
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



def plotImp(model, X , num = 15):

    feature_imp = pd.DataFrame({'Value':model.feature_importance(),'Feature':X.columns})

    plt.figure(figsize=(9, 4))

    sns.set(font_scale = 1.2)

    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])

    plt.title('LightGBM Features (avg over folds) --- ' + str(target))

    plt.tight_layout()

    plt.show()

    

#plotImp(clf, X_train)
%%time



rcParams["figure.figsize"] = 9, 4



for model,target in enumerate(targets,1):

    print(target)

    evals_result = {}

    y = train_targets_scored[target]

    predictions = np.zeros(test_features.shape[0])

    oof_preds = np.zeros(X.shape[0])

    

    for train_idx, test_idx in skf.split(X, y):

        train_data = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx], categorical_feature=["cp_type","cp_time","cp_dose"])

        val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx], categorical_feature=["cp_type","cp_time","cp_dose"])

        

        clf = lgb.train(params, train_data, 10000, valid_sets = [train_data, val_data], verbose_eval=0, early_stopping_rounds=15, evals_result=evals_result)

        

        oof_preds[test_idx] = clf.predict(X.iloc[test_idx])

        predictions += clf.predict(test_features[features]) / skf.n_splits

        

    submission[target] = predictions

    loss = log_loss(y, oof_preds)

    total_loss += loss

    

    print(f"Model:{model} ==> Losses:{loss:.4f}")

    rcParams["figure.figsize"] = 9, 4

    lgb.plot_metric(evals_result)

    

    print("----------")

    

    plotImp(clf, X)

    

    print("----------")

    del predictions, oof_preds,  y, loss, clf

    gc.collect();
print('Overall mean loss: {:.3f}'.format(total_loss / 206))
submission.head()
submission.to_csv('submission.csv', index=False)