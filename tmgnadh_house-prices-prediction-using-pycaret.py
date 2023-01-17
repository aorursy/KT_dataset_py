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
# run this cell of code to install pycaret

!pip install pycaret
#import regression module

from pycaret.regression import *
#import data

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



train.head()
#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)



train.drop(['Utilities'], axis=1, inplace = True)
# Adding total sqfootage feature 

train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']



train.head()
#setting up

Trail1 = setup(data = train, target = 'SalePrice', silent = True, session_id=119, remove_outliers = True, 

               outliers_threshold = 0.01, transform_target = True, feature_selection = True, 

               categorical_features = ['MSSubClass', 'OverallCond','YrSold','MoSold'])
compare_models(blacklist = ['tr'])
cb = create_model('catboost')
#cb_tuned = tune_model('catboost', optimize = 'mse')
predictions1 = predict_model(cb, data = test)

predictions1.head()
Trail2 = setup(train, target = 'SalePrice', session_id = 120, 

             normalize = True, normalize_method = 'zscore',

             transformation = True, transformation_method = 'yeo-johnson', transform_target = True,

             numeric_features=['OverallQual', 'BsmtFullBath', 'BsmtHalfBath', 

                               'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 

                               'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'PoolArea'],

             categorical_features = ['MSSubClass', 'OverallCond','YrSold','MoSold'],

             ordinal_features= {'ExterQual': ['Fa', 'TA', 'Gd', 'Ex'],

                                'ExterCond' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

                                'BsmtQual' : ['Fa', 'TA', 'Gd', 'Ex'], 

                                'BsmtCond' : ['Po', 'Fa', 'TA', 'Gd'],

                                'BsmtExposure' : ['No', 'Mn', 'Av', 'Gd'],

                                'HeatingQC' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

                                'KitchenQual' : ['Fa', 'TA', 'Gd', 'Ex'],

                                'FireplaceQu' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

                                'GarageQual' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

                                'GarageCond' : ['Po', 'Fa', 'TA', 'Gd', 'Ex'],

                                'PoolQC' : ['Fa', 'Gd', 'Ex']},

             polynomial_features = True, trigonometry_features = True, remove_outliers = True, outliers_threshold = 0.01,

             silent = True #silent is set to True for unattended run during kernel execution

             )
compare_models(blacklist = ['tr'])
hr = create_model('huber')
tuned_hr = tune_model('huber', optimize = 'mae')
predictions2 = predict_model(tuned_hr, data = test)

predictions2.head()
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = predictions2['Label']

sub.to_csv('submission.csv',index=False)