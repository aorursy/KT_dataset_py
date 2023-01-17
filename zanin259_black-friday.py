# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# We read in the train and test data

train=pd.read_csv('/kaggle/input/black-friday/train.csv')



test=pd.read_csv('/kaggle/input/black-friday/test.csv')
!pip install feature-engine
train.head()
train.info()
# We find the value counts of all the columns with object type

for i in train.columns:

    if (train[i].dtypes=='O')&(i[-3:]!='_ID'):

        print(train[i].value_counts(),'\n')
# We countplot the gender and marital status to find the purchase.

train['G_M_Combined']=train.apply(lambda x : '%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)



plt.figure(figsize=(9,12))



sns.countplot(train['Age'],hue=train['G_M_Combined'])     
# We find the distribution plot of the city category with respect to purchase

g= sns.FacetGrid(train, col= 'City_Category', sharey= True, aspect= 1.2)

g.map(sns.kdeplot, 'Purchase')
# We perform the barplot for the various age and Gender category

plt.figure(figsize=(12,9))

sns.barplot(x='Age',y='Purchase',hue='Gender',data=train)
# We perform the barplot for the various occupation and Gender category

plt.figure(figsize=(12,9))

sns.barplot(x='Occupation',y='Purchase',hue='Gender',data=train)
# We perform the barplot for the various City_catergory and Gender category

plt.figure(figsize=(12,9))

sns.barplot(x='City_Category',y='Purchase',hue='Gender',data=train)
train.isnull().mean()
# We drop all the irrelavant features

X=train.drop(columns=['User_ID','Product_ID','Gender','Marital_Status','Purchase'])



y=train['Purchase']
# We split the dataset into train and test set

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# we impute the missing values in product_category_2

from feature_engine.missing_data_imputers import ArbitraryNumberImputer



AI=ArbitraryNumberImputer(arbitrary_number=0,variables=['Product_Category_2','Product_Category_3'])



AI.fit(X_train)



X_train=AI.transform(X_train)



X_test=AI.transform(X_test)
# We discritize the age and stay in cuurent city

from feature_engine.categorical_encoders import OrdinalCategoricalEncoder



OCE=OrdinalCategoricalEncoder(variables=['Age','Stay_In_Current_City_Years','G_M_Combined'])



OCE.fit(X_train,y_train)



X_train=OCE.transform(X_train)



X_test=OCE.transform(X_test)
X_train
X_test
# We perform one hot encoding for all the features

from feature_engine.categorical_encoders import OneHotCategoricalEncoder



ohce = OneHotCategoricalEncoder(drop_last=True)



ohce.fit(X_train)



X_train=ohce.transform(X_train)



X_test=ohce.transform(X_test)
# Scaling the values of all the features

from sklearn.preprocessing import StandardScaler



sc=StandardScaler()



sc.fit(X_train)



X_train=sc.transform(X_train)



X_test=sc.transform(X_test)
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor
# We perform the Descision Tree Regression

regressor_DT = DecisionTreeRegressor()



regressor_DT.fit(X_train,y_train)



y_pred_DT=regressor_DT.predict(X_test)
# to find the accuracy score of the values we use mean squared and r2 score

from sklearn.metrics import mean_squared_error,r2_score



mse_DT=mean_squared_error(y_pred_DT,y_test)



r2_DT=r2_score(y_pred_DT,y_test)



print('Mean squared error is {}\nr2 score is {}.'.format(np.sqrt(mse_DT),r2_DT))
# We perform the Random forest regressor

regressor_RF = RandomForestRegressor()



regressor_RF.fit(X_train,y_train)



y_pred_RF=regressor_RF.predict(X_test)
# to find the accuracy score of the values we use mean squared and r2 score



mse_RF=mean_squared_error(y_pred_RF,y_test)



r2_RF=r2_score(y_pred_RF,y_test)



print('Mean squared error is {}\nr2 score is {}.'.format(np.sqrt(mse_RF),r2_RF))
# We perform the gradient boosting regressor

regressor_gb = GradientBoostingRegressor()



regressor_gb.fit(X_train,y_train)



y_pred_gb=regressor_gb.predict(X_test)
# to find the accuracy score of the values we use mean squared and r2 score

mse_gb=mean_squared_error(y_pred_gb,y_test)



r2_gb=r2_score(y_pred_gb,y_test)



print('Mean squared error is {}\nr2 score is {}.'.format(np.sqrt(mse_gb),r2_gb))
test.head()
test.isnull().mean()
# We perform all the imputation in the test data

test['G_M_Combined']=test.apply(lambda x : '%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)



test=test.drop(columns=['User_ID','Product_ID','Gender','Marital_Status'])
# We performt the arbitrary value imputation

test=AI.transform(test)
# we perform ordinal encoding

test=OCE.transform(test)
# We perform the one hot encoding

test=ohce.transform(test)
# Scaling all the values in the features

test=sc.transform(test)
y_pred_test=regressor_gb.predict(test)