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
import lightgbm as lgb

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import cross_validate,StratifiedKFold,cross_val_score,train_test_split

from sklearn.metrics import make_scorer,roc_auc_score

import xgboost as xgb

from catboost import CatBoostClassifier,Pool,CatBoost

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
def met_auc(y_test,y_pred):

    return roc_auc_score(y_test,y_pred)
stratifiedkfold = StratifiedKFold(n_splits = 3)

score_func = {'auc':make_scorer(met_auc)}
train = pd.read_csv('../input/1056lab-fraud-detection-in-credit-card/train.csv').drop('ID',axis=1)
test = pd.read_csv('../input/1056lab-fraud-detection-in-credit-card/test.csv').drop('ID',axis=1)
Y = train['Class'].values

X = train.drop('Class',axis=1).values
X_test_d = test.values
sm = SMOTE(kind='regular')

X_res,Y_res = sm.fit_sample(X,Y)
print(X_res.shape)

print(X_test_d.shape)
gousei = np.concatenate([X_res,X_test_d])

print(gousei.shape)
pca = PCA(n_components=2)

pca.fit(gousei)

gousei_pca = pca.transform(gousei)

print('Original shape: {}'.format(str(gousei.shape)))

print('Reduced shape: {}'.format(str(gousei_pca.shape)))
X_pca,X_test_pca = np.split(gousei_pca,[395964])
X_train, X_test, y_train, y_test = train_test_split(X_pca, Y_res,test_size=0.1,shuffle=True,random_state=42,stratify=Y_res)

train_pool = Pool(X_train,label = y_train)

test_pool = Pool(X_test,label = y_test)
params = {

    'loss_function':'Logloss',

    'num_boost_round':10000,

    'early_stopping_rounds':10

}
model = CatBoost(params)

model.fit(train_pool,eval_set = [test_pool])
history = model.get_evals_result()
train_metric = history['learn']['Logloss']

plt.plot(train_metric, label='train metric')

eval_metric = history['validation']['Logloss']

plt.plot(eval_metric, label='eval metric')



plt.legend()

plt.grid()

plt.show()
X_test_pca
X_test_pool = Pool(X_test_pca)

pred = model.predict(X_test_pool,prediction_type = 'Probability')
sample = pd.read_csv('../input/1056lab-fraud-detection-in-credit-card/sampleSubmission.csv',index_col = 0)
sample['Class'] = pred[:,1]
sample.to_csv('predict_catboost1.csv',header = True)