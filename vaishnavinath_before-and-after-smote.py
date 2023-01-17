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
import numpy as np

import matplotlib.pyplot as plt

import pandas

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

#from catboost import CatBoostClassifier,Pool

from IPython.display import display

import matplotlib.patches as patch

import matplotlib.pyplot as plt

from sklearn.svm import NuSVR

from scipy.stats import norm

from sklearn import svm

import lightgbm as lgb

import xgboost as xgb

import seaborn as sns

import pandas as pd

import numpy as np

import warnings

import time

import glob

import sys

import os

import gc

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
train = pd.read_csv('../input/train.csv')
train.head()
train.shape
train.columns
train.describe()
def check_missing_data(df):

    flag=df.isna().sum().any()

    if flag==True:

        total = df.isnull().sum()

        percent = (df.isnull().sum())/(df.isnull().count()*100)

        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

        data_type = []

        for col in df.columns:

            dtype = str(df[col].dtype)

            data_type.append(dtype)

        output['Types'] = data_type

        return(np.transpose(output))

    else:

        return(False)
check_missing_data(train)
train['target'].value_counts()
def check_balance(df,target):

    check=[]

    print('size of data is:',df.shape[0] )

    for i in [0,1]:

        print('for target  {} ='.format(i))

        print(df[target].value_counts()[i]/df.shape[0]*100,'%')
check_balance(train,'target')
cols=["target","ID_code"]

X = train.drop(cols,axis=1)

y = train["target"]
train_X, val_X, train_y, val_y = train_test_split(X, y,test_size=0.3, random_state=1)

rfc_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
y_pred_rfc=rfc_model.predict(val_X)
print("Accuracy:",metrics.accuracy_score(val_y, y_pred_rfc))

print("precision:",metrics.precision_score(val_y, y_pred_rfc))
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
y_pred_tree = tree_model.predict(val_X)
print("Accuracy:",metrics.accuracy_score(val_y, y_pred_tree))

print("precision:",metrics.precision_score(val_y, y_pred_tree))
logreg = LogisticRegression()

logreg.fit(train_X,train_y)

y_pred=logreg.predict(val_X)
print("Accuracy:",metrics.accuracy_score(val_y, y_pred))

print("Precision:",metrics.precision_score(val_y, y_pred))
params = {'num_leaves': 9,

         'min_data_in_leaf': 42,

         'objective': 'binary',

         'max_depth': 16,

         'learning_rate': 0.0123,

         'boosting': 'gbdt',

         'bagging_freq': 5,

         'bagging_fraction': 0.8,

         'feature_fraction': 0.8201,

         'bagging_seed': 11,

         'reg_alpha': 1.728910519108444,

         'reg_lambda': 4.9847051755586085,

         'random_state': 42,

         'metric': 'auc',

         'verbosity': -1,

         'subsample': 0.81,

         'min_gain_to_split': 0.01077313523861969,

         'min_child_weight': 19.428902804238373,

         'num_threads': 4}
fold_n=2

folds = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=10)

%matplotlib inline

%precision 4

warnings.filterwarnings('ignore')

plt.style.use('ggplot')

np.set_printoptions(suppress=True)

pd.set_option("display.precision", 15)
y_pred_lgb = np.zeros(len(val_X))

for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):

    print('Fold', fold_n, 'started at', time.ctime())

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    train_data = lgb.Dataset(X_train, label=y_train)

    valid_data = lgb.Dataset(X_valid, label=y_valid)

        

    lgb_model = lgb.train(params,train_data,num_boost_round=2000,#change 20 to 2000

                    valid_sets = [train_data, valid_data],verbose_eval=300,early_stopping_rounds = 200)##change 10 to 200

            

    y_pred_lgb += lgb_model.predict(val_X, num_iteration=lgb_model.best_iteration)/5
model = XGBClassifier().fit(X_train, y_train)

y_pred = model.predict(val_X)
print("Accuracy:",metrics.accuracy_score(val_y, y_pred))

print("precision:",metrics.precision_score(val_y, y_pred))
from imblearn.over_sampling import SMOTE

from collections import Counter

sm = SMOTE(random_state=42)

X_resamp_tr, y_resamp_tr = sm.fit_resample(X, y)

print('Resampled dataset shape %s' % Counter(y_resamp_tr))

X_resamp_tr = pandas.DataFrame(X_resamp_tr)

y_resamp_tr = pandas.DataFrame({"target": y_resamp_tr})
X_resamp_tr.head()
y_resamp_tr.head()
train_X, val_X, train_y, val_y = train_test_split(X_resamp_tr, y_resamp_tr,test_size=0.3, random_state=1)
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred=logreg.predict(val_X)
print("Accuracy:",metrics.accuracy_score(val_y, y_pred))

print("Precision:",metrics.precision_score(val_y, y_pred))
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
y_pred_tree = tree_model.predict(val_X)
print("Accuracy:",metrics.accuracy_score(val_y, y_pred_tree))

print("precision:",metrics.precision_score(val_y, y_pred_tree))
rfc_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
y_pred_rfc=rfc_model.predict(val_X)
print("Accuracy:",metrics.accuracy_score(val_y, y_pred_rfc))

print("precision:",metrics.precision_score(val_y, y_pred_rfc))
y_pred_lgb = np.zeros(len(val_X))

for fold_n, (train_index, valid_index) in enumerate(folds.split(X_resamp_tr,y_resamp_tr)):

    print('Fold', fold_n, 'started at', time.ctime())

    X_train, X_valid = X_resamp_tr.iloc[train_index], X_resamp_tr.iloc[valid_index]

    y_train, y_valid = y_resamp_tr.iloc[train_index], y_resamp_tr.iloc[valid_index]

    

    train_data = lgb.Dataset(X_train, label=y_train)

    valid_data = lgb.Dataset(X_valid, label=y_valid)

        

    lgb_model = lgb.train(params,train_data,num_boost_round=2000,#change 20 to 2000

                    valid_sets = [train_data, valid_data],verbose_eval=300,early_stopping_rounds = 200)##change 10 to 200

            

    y_pred_lgb += lgb_model.predict(val_X, num_iteration=lgb_model.best_iteration)/5
model = XGBClassifier().fit(X_train, y_train)

y_pred = model.predict(val_X)
print("Accuracy:",metrics.accuracy_score(val_y, y_pred))

print("precision:",metrics.precision_score(val_y, y_pred))