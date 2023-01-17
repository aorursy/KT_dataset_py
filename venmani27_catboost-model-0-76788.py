# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
train.head()
train.sort_index(inplace=True)

train_y = train['target']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)
from sklearn.metrics import roc_auc_score

cat_feat_to_encode = train.columns.tolist();  smoothing=0.15

import category_encoders as ce

oof = pd.DataFrame([])

df=train
df
df['bin_0']=df['bin_0'].fillna('0')

df['bin_1']=df['bin_1'].fillna('0')

df['bin_2']=df['bin_2'].fillna('0')

df['bin_3']=df['bin_3'].fillna('F')

df['bin_4']=df['bin_4'].fillna('N')

df['nom_0']=df['nom_0'].fillna('Red')

df['nom_1']=df['nom_1'].fillna('Triangle')

df['nom_2']=df['nom_2'].fillna('Hamster')

df['nom_3']=df['nom_3'].fillna('India')

df['nom_4']=df['nom_4'].fillna('Theremin')

df['nom_5']=df['nom_5'].fillna('None')

df['nom_6']=df['nom_6'].fillna('None')

df['nom_7']=df['nom_7'].fillna('None')

df['nom_8']=df['nom_8'].fillna('None')

df['nom_9']=df['nom_9'].fillna('None')

df['ord_0']=df['ord_0'].fillna('1')

df['ord_1']=df['ord_1'].fillna('Novice')

df['ord_2']=df['ord_2'].fillna('Freezing')

df['ord_3']=df['ord_3'].fillna('n')

df['ord_4']=df['ord_4'].fillna('N')

df['ord_5']=df['ord_5'].fillna('Fl')

df['day']=df['day'].fillna('4')

df['month']=df['month'].fillna('6')

test['bin_0']=test['bin_0'].fillna('0')

test['bin_1']=test['bin_1'].fillna('0')

test['bin_2']=test['bin_2'].fillna('0')

test['bin_3']=test['bin_3'].fillna('F')

test['bin_4']=test['bin_4'].fillna('N')

test['nom_0']=test['nom_0'].fillna('Red')

test['nom_1']=test['nom_1'].fillna('Triangle')

test['nom_2']=test['nom_2'].fillna('Hamster')

test['nom_3']=test['nom_3'].fillna('India')

test['nom_4']=test['nom_4'].fillna('Theremin')

test['nom_5']=test['nom_5'].fillna('None')

test['nom_6']=test['nom_6'].fillna('None')

test['nom_7']=test['nom_7'].fillna('None')

test['nom_8']=test['nom_8'].fillna('None')

test['nom_9']=test['nom_9'].fillna('None')

test['ord_0']=test['ord_0'].fillna('1')

test['ord_1']=test['ord_1'].fillna('Novice')

test['ord_2']=test['ord_2'].fillna('Freezing')

test['ord_3']=test['ord_3'].fillna('n')

test['ord_4']=test['ord_4'].fillna('N')

test['ord_5']=test['ord_5'].fillna('Fl')

test['day']=test['day'].fillna('4')

test['month']=test['month'].fillna('6')
from sklearn.model_selection import StratifiedKFold

for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=101, shuffle=True).split(df, train_y):

    target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

    target_encoder.fit(df.iloc[tr_idx, :], train_y.iloc[tr_idx])

    oof = oof.append(target_encoder.transform(df.iloc[oof_idx, :]), ignore_index=False)

target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

target_encoder.fit(df, train_y)

df = oof.sort_index()

test = target_encoder.transform(test)
df.head()
from catboost import CatBoostClassifier

clf = CatBoostClassifier(iterations=1000,

                              learning_rate=0.15,

                              depth=10,

                              bootstrap_type='Bernoulli',

                              loss_function='Logloss',

                              subsample=0.99,

                              eval_metric='AUC',

                              metric_period=20,

                              allow_writing_files=False)

clf.fit(

        df, train_y,

        verbose_eval=100, 

        early_stopping_rounds=50,

        eval_set=(df, train_y),

        use_best_model=False,

        plot=True

)
pd.DataFrame({'id': test_id, 'target': clf.predict_proba(test)[:,1]}).to_csv('catsubmission4.csv', index=False)