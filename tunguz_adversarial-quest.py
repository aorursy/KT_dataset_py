import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn import model_selection, preprocessing, metrics

from sklearn.linear_model import LogisticRegression

from sklearn.impute import SimpleImputer

from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import shap

import os

print(os.listdir("../input"))

from sklearn import preprocessing

import xgboost as xgb

import gc





import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%%time

train = np.load('../input/distilbert-use-features-just-the-features/X_train.npy')

test = np.load('../input/distilbert-use-features-just-the-features/X_test.npy')

train.shape
test.shape
np.zeros((6079,))
train_test = np.vstack([train, test])

target = np.hstack([np.zeros((6079,)), np.ones((476,))])

del train, test

gc.collect()
train, test, train_y, test_y = model_selection.train_test_split(train_test, target, test_size=0.33, random_state=42, shuffle=True)

del train_test, target

gc.collect()
train = lgb.Dataset(train, label=train_y)

test = lgb.Dataset(test, label=test_y)

gc.collect()
param = {'num_leaves': 50,

         'min_data_in_leaf': 20, 

         'objective':'binary',

         'max_depth': 2,

         'learning_rate': 0.01,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.5,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 44,

         "metric": 'auc',

         "verbosity": -1}
num_round = 500

clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=1000, early_stopping_rounds = 1000)
features = ['feature_'+str(x) for x in range(3142)]



feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])



plt.figure(figsize=(20, 20))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(100))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()

plt.savefig('lgbm_importances-01.png')