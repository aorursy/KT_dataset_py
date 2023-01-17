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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

description = open("../input/house-prices-advanced-regression-techniques/data_description.txt",'r')
print(description.read())
train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_df.head(5)
test_df.head(5)
train_df.info()
test_df.info()
train_df.describe()
matrix_corr = train_df.corr()['SalePrice']
matrix_corr.sort_values(ascending=False)
train_df.hist(bins=50,figsize=(25,20))
plt.show()
train_df.plot(kind='scatter',x='OverallQual',y='SalePrice')
train_df.plot(kind='scatter',x='GrLivArea',y='SalePrice')
train_df.plot(kind='scatter',x='GarageCars',y='SalePrice')
train_df.dtypes.value_counts()
test_df.dtypes.value_counts()
object_df = train_df.select_dtypes(include=[np.object])
object_df
train_df.replace('Ex',5,inplace=True)
train_df.replace('Gd',4,inplace=True)
train_df.replace('TA',3,inplace=True)
train_df.replace('Fa',2,inplace=True)
train_df.replace('Po',1,inplace=True)
train_df.replace('None',0,inplace=True)
train_df.replace('GLQ',6,inplace=True)
train_df.replace('ALQ',5,inplace=True)
train_df.replace('BLQ',4,inplace=True)
train_df.replace('Rec',3,inplace=True)
train_df.replace('LNQ',2,inplace=True)
train_df.replace('UNF',1,inplace=True)
train_df.replace('Y',1,inplace=True)
train_df.replace('P',1,inplace=True)
train_df.replace('N',0,inplace=True)
#TESTING 
test_df.replace('Ex',10,inplace=True)
test_df.replace('Gd',9,inplace=True)
test_df.replace('TA',8,inplace=True)
test_df.replace('Fa',7,inplace=True)
test_df.replace('Po',6,inplace=True)
test_df.replace('None',0,inplace=True)
test_df.replace('GLQ',6,inplace=True)
test_df.replace('ALQ',5,inplace=True)
test_df.replace('BLQ',4,inplace=True)
test_df.replace('Rec',3,inplace=True)
test_df.replace('LNQ',2,inplace=True)
test_df.replace('UNF',1,inplace=True)
test_df.replace('Y',1,inplace=True)
test_df.replace('P',1,inplace=True)
test_df.replace('N',0,inplace=True)
train_df.dtypes.value_counts()
test_df.dtypes.value_counts()
train_df.isnull().sum().sort_values(ascending=False).iloc[:15]
test_df.isnull().sum().sort_values(ascending=False).iloc[:12]
print('For training dataset')
print('Null Val. Percentage in PoolQC:',(1453/len(train_df['PoolQC']))*100)
print('Null Val. Percentage in MiscFeature:',(1406/len(train_df['MiscFeature']))*100)
print('Null Val. Percentage in Alley:',(1369/len(train_df['Alley']))*100)
print('Null Val. Percentage in Fence:',(1179/len(train_df['Fence']))*100)
print('Null Val. Percentage in FireplaceQu:',(690/len(train_df['FireplaceQu']))*100)
print('Null Val. Percentage in LotFrontage:',(259/len(train_df['LotFrontage']))*100)
print('\n')
print('For testing dataset')
print('Null Val. Percentage in PoolQC:',(1456/len(test_df['PoolQC']))*100)
print('Null Val. Percentage in MiscFeature:',(1408/len(test_df['MiscFeature']))*100)
print('Null Val. Percentage in Alley:',(1352/len(test_df['Alley']))*100)
print('Null Val. Percentage in Fence:',(1169/len(test_df['Fence']))*100)
print('Null Val. Percentage in FireplaceQu:',(730/len(test_df['FireplaceQu']))*100)
print('Null Val. Percentage in LotFrontage:',(227/len(test_df['LotFrontage']))*100)





train_df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1,inplace=True)
test_df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1,inplace=True)
train_numerical = train_df.select_dtypes(exclude=[np.object])
test_numerical = test_df.select_dtypes(exclude=[np.object])
train_object = train_df.select_dtypes(include=[np.object])
test_object = test_df.select_dtypes(include=[np.object])
train_numerical.isnull().sum().sort_values(ascending = False).iloc[:6]
test_numerical.isnull().sum().sort_values(ascending = False).iloc[:10]
train_numerical['LotFrontage'].fillna(train_numerical['LotFrontage'].median(),inplace=True)
train_numerical['GarageQual'].fillna(train_numerical['GarageQual'].median(),inplace=True)
train_numerical['GarageYrBlt'].fillna(train_numerical['GarageYrBlt'].median(),inplace=True)
train_numerical['BsmtQual'].fillna(train_numerical['BsmtQual'].median(),inplace=True)
train_numerical['BsmtCond'].fillna(train_numerical['BsmtCond'].median(),inplace=True)
train_numerical['MasVnrArea'].fillna(train_numerical['MasVnrArea'].median(),inplace=True)

test_numerical['LotFrontage'].fillna(test_numerical['LotFrontage'].median(),inplace=True)
test_numerical['GarageYrBlt'].fillna(test_numerical['GarageYrBlt'].median(),inplace=True)
test_numerical['GarageCond'].fillna(test_numerical['GarageCond'].median(),inplace=True)
test_numerical['GarageArea'].fillna(test_numerical['GarageArea'].median(),inplace=True)
test_numerical['MasVnrArea'].fillna(test_numerical['MasVnrArea'].median(),inplace=True)
test_numerical['BsmtCond'].fillna(test_numerical['BsmtCond'].median(),inplace=True)
test_numerical['BsmtQual'].fillna(test_numerical['BsmtQual'].median(),inplace=True)
train_numerical.fillna(0,inplace=True)
test_numerical.fillna(0,inplace=True)
sns.heatmap(train_numerical.isnull())
sns.heatmap(test_numerical.isnull())
train_object = pd.get_dummies(train_object)
test_object = pd.get_dummies(test_object)
train_df_updated = pd.concat([train_numerical,train_object],axis=1)
test_df_updated = pd.concat([test_numerical,test_object],axis=1)
X = train_df_updated.drop(['SalePrice','Id'],axis=1)
y = train_df['SalePrice']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
lr = LinearRegression()
dcr = DecisionTreeRegressor()
rfr = RandomForestRegressor()
svr = SVR()

lr.fit(X_train,y_train)
dcr.fit(X_train,y_train)
rfr.fit(X_train,y_train)
svr.fit(X_train,y_train)
prediction = lr.predict(X_test)
predict1 = dcr.predict(X_test)
predict2 = rfr.predict(X_test)
predict3 = svr.predict(X_test)
from sklearn.metrics import mean_squared_error,r2_score
import math
print('RMSE for Random Forest:',math.sqrt(mean_squared_error(y_test,predict2)))
print('r2 score for Random Forest:',r2_score(y_test,predict2))
print('\n')
print('RMSE for Support Vector Regressor:',math.sqrt(mean_squared_error(y_test,predict3)))
print('r2 score for Support Vector Regressor:',r2_score(y_test,predict3))
print('\n')
print('RMSE for Decision Tree:',math.sqrt(mean_squared_error(y_test,predict1)))
print('r2 score for Decision Tree:',r2_score(y_test,predict3))
print('\n')
print('RMSE for Linear Regression:',math.sqrt(mean_squared_error(y_test,prediction)))



