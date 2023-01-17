# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
rawTrainData = pd.read_csv('../input/train.csv')

print(rawTrainData.head(50))
import seaborn as sns

import matplotlib.pyplot as plt

sns.distplot(rawTrainData['target'], kde=False, rug=True);
updatedTrainData = rawTrainData.drop(columns=['target','ID_code'])

target = rawTrainData['target']
corrcoef = np.corrcoef(updatedTrainData, rowvar=0)

print('The determinant value of the correlation matrix is: ',np.linalg.det(corrcoef))

print('The condition number of the correlation matrix is: ',np.linalg.cond(corrcoef))

w,v = np.linalg.eig(corrcoef)

print('The minimum eigen values of the correlation matrix are: ',np.min(w))
features = "+".join(updatedTrainData.columns)

from patsy import dmatrices

y, X = dmatrices('target ~' + features, rawTrainData, return_type='dataframe')
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from tqdm import tqdm



vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in tqdm(range(X.shape[1]))]

vif["features"] = X.columns
vif.round(1)

#All the variables seem to independent from the eigen values, the determinant factor and the VIF
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaledTrainData = scaler.fit_transform(updatedTrainData)
#try a basic logistic regression model without any class weights

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr',C=0.5)
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

X_train,X_val,Y_train,Y_val = train_test_split(scaledTrainData,target,test_size = 0.25)

clf.fit(X_train,Y_train)

Y_predict = clf.predict(X_val)

print(f1_score(Y_val,Y_predict))
from sklearn.metrics import roc_auc_score

print(roc_auc_score(Y_val,Y_predict))
clf_balanced = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr',C=0.5,class_weight = 'balanced')

clf_balanced.fit(X_train,Y_train)

Y_predict_balanced = clf_balanced.predict(X_val)

print(f1_score(Y_val,Y_predict_balanced))
print(roc_auc_score(Y_val,Y_predict_balanced))
# a penalized logistic regression model works better than basic logistic regression model.

# So penalized models might be worth looking in to.
parameters = {'C':[0.25, 0.5, 0.75],'class_weight':['balanced', {0:.1,1:.9},{0:.05,1:.95},{0:.01,1:.99},{0:.2,1:.8}]}

logit = LogisticRegression(random_state=0,solver='lbfgs',multi_class='ovr')

clf_grid = GridSearchCV(logit, parameters, cv=5,scoring = 'f1')

clf_grid.fit(scaledTrainData,target)
print(clf_grid.best_params_)

print(clf_grid.best_score_)
#import lightgbm as lgb

#param = {'num_leaves':31, 'num_trees':100, 'objective':'binary'}

#param['metric'] = 'auc'

#num_round = 10

#train_data = lgb.Dataset(X_train, label=Y_train)

#validation_data = lgb.Dataset(X_val, label=Y_val)

#bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
import lightgbm as lgb

parameters = {'num_leaves':[16,31,64],'n_estimators':[100,200],'learning_rate':[0.05,0.1,0.5]}

lgbm_clf = lgb.LGBMClassifier(boosting_type = 'gbdt',objective = 'binary')

lgbm_grid = GridSearchCV(lgbm_clf, parameters, verbose=2, cv=5, scoring='roc_auc')

lgbm_grid.fit(scaledTrainData,target)
print(lgbm_grid.best_params_)

print(lgbm_grid.best_score_)
rawTestData = pd.read_csv('../input/test.csv')

updatedTestData = rawTestData.drop(columns=['ID_code'])

scaledTestData = scaler.transform(updatedTestData)

Y_test = lgbm_grid.predict_proba(scaledTestData)
sns.distplot(Y_test, kde=False, rug=True);
sampleData = pd.read_csv('../input/sample_submission.csv')

ID_code = sampleData["ID_code"]

prediction = pd.DataFrame(Y_test, index= ID_code)

prediction.columns = ["target"]

prediction.index.name = "ID_code"

prediction.head()
prediction.to_csv("sample_submission.csv")