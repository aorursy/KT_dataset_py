# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



y = train_data['SalePrice']

X = train_data.drop('SalePrice', axis =1)



objcols = list(X.select_dtypes(include = 'object').columns)

label_encoder = LabelEncoder()

label_X = X



for col in objcols:

    label_X[col] = label_encoder.fit_transform(X[col].astype(str))    

    

my_imputer = SimpleImputer(strategy='most_frequent')



X = pd.DataFrame(my_imputer.fit_transform(label_X))

train_x, val_x, train_y , val_y = train_test_split(X,y,random_state = 1)



from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error



xg_model = XGBRegressor(n_estimators = 500 , learning_rate = 0.04,random_state = 1)

xg_model.fit(train_x, train_y, early_stopping_rounds = 5, eval_set=[(val_x, val_y)],verbose = False)

pred_val = xg_model.predict(val_x)

mae = mean_absolute_error(val_y, pred_val)

mae



xg_model_full =  XGBRegressor(n_estimators = 500 , learning_rate = 0.02,random_state = 1)

xg_model_full.fit(X,y)
label_test_x = test_data



for col in objcols:

    label_test_x[col] = label_encoder.fit_transform(label_test_x[col].astype(str))



test_x = pd.DataFrame(my_imputer.transform(label_test_x))



Id = test_data['Id']



SalePrice = xg_model_full.predict(test_x)



dic = {'Id':Id, 'SalePrice': SalePrice}

df = pd.DataFrame(dic)

df.to_csv('submission.csv', index = False)

df