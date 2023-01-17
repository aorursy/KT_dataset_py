!pip install pycaret==1.0.0
import numpy as np 

import pandas as pd 

import sklearn

from sklearn.model_selection import train_test_split
dataset=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

dataset.head()
dataset.isna().sum()
train,test=train_test_split(dataset,test_size=0.2,random_state=42)
print('train shape:',train.shape ,'test shape:',test.shape)
from pycaret.classification import *
train.head()
train.describe()
clf=setup(data=train,

         target='Outcome',

        numeric_imputation = 'mean',

         silent = True)
compare_models()
lightgbm  = create_model('lightgbm')



## Here this model automatically use 10-Fold CV.
tuned_lightgbm = tune_model('lightgbm')
evaluate_model(tuned_lightgbm)
plot_model(estimator = tuned_lightgbm, plot = 'learning')
plot_model(estimator = tuned_lightgbm, plot = 'auc')
plot_model(estimator = tuned_lightgbm, plot = 'confusion_matrix')
plot_model(estimator = tuned_lightgbm, plot = 'feature')
interpret_model(tuned_lightgbm)
logr  = create_model('lr');      

xgb   = create_model('xgboost');            



#blending 3 models

blend = blend_models(estimator_list=[tuned_lightgbm,logr,xgb])
from pycaret.regression import *
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train.describe()
categorical_col_list=[i for i in train.columns if train[i].dtypes =='object']

categorical_col_list
reg = setup(data = train, 

             target = 'SalePrice',

             numeric_imputation = 'mean',

             categorical_features = categorical_col_list, 

             ignore_features = ['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','Utilities'],

             normalize = True,

             silent = True)
compare_models()
cb = create_model('catboost')
tuned_cb = tune_model('catboost')
interpret_model(tuned_cb)