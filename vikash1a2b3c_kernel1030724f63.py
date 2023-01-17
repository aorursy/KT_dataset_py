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
# # # # # # # # # #  CATBOOST MODEL # # # # # # # # 



from catboost import CatBoostRegressor



#Read trainig and testing files

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")









#Imputing missing values for both train and test

train.fillna(-999, inplace=True)

test.fillna(-999,inplace=True)



#Creating a training set for modeling and validation set to check model performance

train_id=train.Id

X = train.drop(['SalePrice','Id'], axis=1)

y = train.SalePrice



from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.7, random_state=1234)



#Look at the data type of variables

X.dtypes



#Now, you’ll see that we will only identify categorical variables. We will not perform any preprocessing steps for categorical variables:

categorical_features_indices = np.where(X.dtypes != np.float)[0]



#importing library and building model

model=CatBoostRegressor(iterations=800, depth=1, learning_rate=0.01, loss_function='RMSE')

model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)
