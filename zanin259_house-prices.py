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

train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
!pip install feature-engine
train.head(10)
test.select_dtypes(exclude='object').head()
train.isnull().mean()
test.select_dtypes(exclude='object').isnull().mean()
test.select_dtypes(include='object').isnull().mean()
# We find columns with NAN for numerical values

train.select_dtypes(exclude='object').head()
train.select_dtypes(exclude='object').isnull().mean()
train.select_dtypes(include='object').isnull().mean()
# We find the number of unique values in all the categorical varible

for col in train.columns:

    if(train[col].dtypes=='O')&(col!='Id'):

        print(col,':',train[col].nunique())
# We do the same for the object column

train.select_dtypes(include='object').isnull().mean()
# We plot a histogram for all the numerical values

train.hist(figsize=(20,20))
# We split the data into X and y to create the train and test set

X=train.drop(columns='SalePrice')



y=train['SalePrice']



from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# We fill the missing nan values in numerical features



from feature_engine.missing_data_imputers import ArbitraryNumberImputer,MeanMedianImputer



AT=ArbitraryNumberImputer(arbitrary_number=0,variables='MasVnrArea')



AT.fit(X_train)



X_train=AT.transform(X_train)

X_test=AT.transform(X_test)
MI = MeanMedianImputer(imputation_method='mean')



MI.fit(X_train)



X_train=MI.transform(X_train)



X_test=MI.transform(X_test)
# We do the same for test set and fill the NAN Values

test=AT.transform(test)



test=MI.transform(test)
# We convert the columns to integer

X_train['LotFrontage']=X_train['LotFrontage'].astype('int')

X_train['MasVnrArea']=X_train['MasVnrArea'].astype('int')

X_train['GarageYrBlt']=X_train['GarageYrBlt'].astype('int')



X_test['LotFrontage']=X_test['LotFrontage'].astype('int')

X_test['MasVnrArea']=X_test['MasVnrArea'].astype('int')

X_test['GarageYrBlt']=X_test['GarageYrBlt'].astype('int')
# We check if the pool value has an impact on the saleprice

sns.kdeplot(train.loc[train['PoolQC'].notna(),'SalePrice'],label='With Pool')

sns.kdeplot(train.loc[train['PoolQC'].isna(),'SalePrice'],label='Without Pool')
# We do the same in the case of alley having an impact on the saleprice

sns.kdeplot(train.loc[train['Alley'].notna(),'SalePrice'])

sns.kdeplot(train.loc[train['Alley'].isna(),'SalePrice'])
# We do the same for the miscfeature

sns.kdeplot(train.loc[train['MiscFeature'].notna(),'SalePrice'],label='with misc_features')

sns.kdeplot(train.loc[train['MiscFeature'].isna(),'SalePrice'],label='without_feature')
sns.kdeplot(train.loc[train['Fence'].notna(),'SalePrice'],label='with fence')

sns.kdeplot(train.loc[train['Fence'].isna(),'SalePrice'],label='without fence')
# We fill the NAN values in Categorical Variables.



from feature_engine.missing_data_imputers import CategoricalVariableImputer



CI = CategoricalVariableImputer(variables=['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature'])



CI.fit(X_train)



X_train=CI.transform(X_train)



X_test=CI.transform(X_test)
from feature_engine.missing_data_imputers import FrequentCategoryImputer



FI = FrequentCategoryImputer()



FI.fit(X_train)



X_train=FI.transform(X_train)



X_test=FI.transform(X_test)
# We do the same steps for the test set



test=CI.transform(test)

test=FI.transform(test)
# We create a new feature known as House Life, it is the time period within which the house was sold

X_train['House Life']=np.where(X_train['YearRemodAdd']>X_train['YearBuilt'],X_train['YrSold']-X_train['YearRemodAdd'],X_train['YrSold']-X_train['YearBuilt'])



X_test['House Life']=np.where(X_test['YearRemodAdd']>X_test['YearBuilt'],X_test['YrSold']-X_test['YearRemodAdd'],X_test['YrSold']-X_test['YearBuilt'])
# We perform the same for the test data



test['House Life']=np.where(test['YearRemodAdd']>test['YearBuilt'],test['YrSold']-test['YearRemodAdd'],test['YrSold']-test['YearBuilt'])
X_train.select_dtypes(exclude='object')
# We discritize the the month sold and garage year built

from feature_engine.discretisers import EqualWidthDiscretiser



ED = EqualWidthDiscretiser(bins=10,variables=['MoSold','GarageYrBlt'])



ED.fit(X_train)



X_train=ED.transform(X_train)



X_test=ED.transform(X_test)
# We do the same in the test set 

test=ED.transform(test)
#  We create a new feature where we find if the house was remodled

X_train['Remodled House']=np.where(X_train['YearRemodAdd']>X_train['YearBuilt'],1,0)



X_test['Remodled House']=np.where(X_test['YearRemodAdd']>X_test['YearBuilt'],1,0)
# We perform the same thing in the test dataset



test['Remodled House']=np.where(test['YearRemodAdd']>test['YearBuilt'],1,0)
# We create a new feature showing New houses

X_train['New Houses']=np.where(X_train['YearBuilt']==X_train['YrSold'],1,0)



X_test['New Houses']=np.where(X_test['YearBuilt']==X_test['YrSold'],1,0)
# We perform the same for test set 



test['New Houses']=np.where(test['YearBuilt']==test['YrSold'],1,0)
qualitative=[col for col in train.columns if train[col].dtypes=='O']
count=1

plt.figure(figsize=(20,10))

for col in qualitative[:9]:

    temp_df = pd.Series(X_train[col].value_counts() / len(X_train) )



    # make plot with the above percentages

    plt.subplot(3,3, count)

    fig = temp_df.sort_values(ascending=False).plot.bar()

    fig.set_xlabel(col)

    

    # add a line at 4 % to flag the threshold for rare categories

    fig.axhline(y=0.04, color='red')

    fig.set_ylabel('Percentage of houses')

    count +=1
count=1

plt.figure(figsize=(20,10))

for col in qualitative[9:18]:

    temp_df = pd.Series(X_train[col].value_counts() / len(X_train) )



    # make plot with the above percentages

    plt.subplot(3,3, count)

    fig = temp_df.sort_values(ascending=False).plot.bar()

    fig.set_xlabel(col)

    

    # add a line at 4 % to flag the threshold for rare categories

    fig.axhline(y=0.04, color='red')

    fig.set_ylabel('Percentage of houses')

    count +=1
count=1

plt.figure(figsize=(20,10))

for col in qualitative[18:27]:

    temp_df = pd.Series(X_train[col].value_counts() / len(X_train) )



    # make plot with the above percentages

    plt.subplot(3,3, count)

    fig = temp_df.sort_values(ascending=False).plot.bar()

    fig.set_xlabel(col)

    

    # add a line at 4 % to flag the threshold for rare categories

    fig.axhline(y=0.04, color='red')

    fig.set_ylabel('Percentage of houses')

    count +=1
count=1

plt.figure(figsize=(20,10))

for col in qualitative[27:36]:

    temp_df = pd.Series(X_train[col].value_counts() / len(X_train) )



    # make plot with the above percentages

    plt.subplot(3,3, count)

    fig = temp_df.sort_values(ascending=False).plot.bar()

    fig.set_xlabel(col)

    

    # add a line at 4 % to flag the threshold for rare categories

    fig.axhline(y=0.04, color='red')

    fig.set_ylabel('Percentage of houses')

    count +=1
count=1

plt.figure(figsize=(20,10))

for col in qualitative[36:44]:

    temp_df = pd.Series(X_train[col].value_counts() / len(X_train) )



    # make plot with the above percentages

    plt.subplot(3,3, count)

    fig = temp_df.sort_values(ascending=False).plot.bar()

    fig.set_xlabel(col)

    

    # add a line at 4 % to flag the threshold for rare categories

    fig.axhline(y=0.04, color='red')

    fig.set_ylabel('Percentage of houses')

    count +=1
# We encoed the rare labels

from feature_engine.categorical_encoders import RareLabelCategoricalEncoder



RE=RareLabelCategoricalEncoder(tol=0.04,variables=['Neighborhood','Condition1','Condition2','HouseStyle','RoofMatl','Exterior1st','Exterior2nd','Heating','Functional','SaleType'])



RE.fit(X_train)



X_train=RE.transform(X_train)



X_test=RE.transform(X_test)
# We perform the same on the test set



test=RE.transform(test)
# We drop the unwanted features

X_train=X_train.drop(columns=['Id','YearBuilt','YearRemodAdd','YrSold','PoolArea','PoolQC','Alley'])

X_test=X_test.drop(columns=['Id','YearBuilt','YearRemodAdd','YrSold','PoolArea','PoolQC','Alley'])
# We perform the same for test dataset

test=test.drop(columns=['Id','YearBuilt','YearRemodAdd','YrSold','PoolArea','PoolQC','Alley'])
X_train.head()
# We create few new features



X_train['Total_sqr_footage'] = (X_train['BsmtFinSF1'] + X_train['BsmtFinSF2'] +

                                 X_train['1stFlrSF'] + X_train['2ndFlrSF'])



X_train['Total_Bathrooms'] = (X_train['FullBath'] + (0.5 * X_train['HalfBath']) +

                               X_train['BsmtFullBath'] + (0.5 * X_train['BsmtHalfBath']))



X_train['Total_porch_sf'] = (X_train['OpenPorchSF'] + X_train['3SsnPorch'] +

                              X_train['EnclosedPorch'] + X_train['ScreenPorch'] +

                              X_train['WoodDeckSF'])
X_test['Total_sqr_footage'] = (X_test['BsmtFinSF1'] + X_test['BsmtFinSF2'] +

                                 X_test['1stFlrSF'] + X_test['2ndFlrSF'])



X_test['Total_Bathrooms'] = (X_test['FullBath'] + (0.5 * X_test['HalfBath']) +

                               X_test['BsmtFullBath'] + (0.5 * X_test['BsmtHalfBath']))



X_test['Total_porch_sf'] = (X_test['OpenPorchSF'] + X_test['3SsnPorch'] +

                              X_test['EnclosedPorch'] + X_test['ScreenPorch'] +

                              X_test['WoodDeckSF'])
test['Total_sqr_footage'] = (test['BsmtFinSF1'] + test['BsmtFinSF2'] +

                                 test['1stFlrSF'] + test['2ndFlrSF'])



test['Total_Bathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) +

                               test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))



test['Total_porch_sf'] = (test['OpenPorchSF'] + test['3SsnPorch'] +

                              test['EnclosedPorch'] + test['ScreenPorch'] +

                              test['WoodDeckSF'])
X_train['TotalSF']=X_train['TotalBsmtSF'] + X_train['1stFlrSF'] + X_train['2ndFlrSF']

X_test['TotalSF']=X_test['TotalBsmtSF'] + X_test['1stFlrSF'] + X_test['2ndFlrSF']
test['TotalSF']=test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

from feature_engine.categorical_encoders import OneHotCategoricalEncoder



ohce=OneHotCategoricalEncoder(drop_last=True)



ohce.fit(X_train)



X_train=ohce.transform(X_train)



X_test=ohce.transform(X_test)
# WE perform the same in the test dataset

test=ohce.transform(test)
# We scale the dataframe to the standard scale

from sklearn.preprocessing import StandardScaler



X_train_cols= X_train.columns



sc=StandardScaler()



sc.fit(X_train)



X_train= pd.DataFrame(sc.transform(X_train), columns= X_train_cols)



X_test= pd.DataFrame(sc.transform(X_test), columns= X_train_cols)
# we do the same in the test data



test= pd.DataFrame(sc.transform(test), columns= X_train_cols)
np.random.seed(seed=42)
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,BaggingRegressor

from xgboost import XGBRegressor
# We the perform the Support Vector Regressor



regressor_svr=SVR()



regressor_svr.fit(X_train,y_train)



y_pred_svr=regressor_svr.predict(X_test)
# we find the accuracy of the prediction using mean squared error, r2 score

from sklearn.metrics import mean_squared_error,r2_score

    

mse_svr=mean_squared_error(y_pred_svr,y_test)



r2_score_svr=r2_score(y_pred_svr,y_test)



print('the mean squared error is {}\nr2 score is {}'.format(np.sqrt(mse_svr),r2_score_svr))
# We use random forest regression

regressor_rf = RandomForestRegressor(random_state=42)



regressor_rf.fit(X_train,y_train)



y_pred_rf=regressor_rf.predict(X_test)
# we find the accuracy of the prediction using mean squared error, r2 score

mse_rf=mean_squared_error(y_pred_rf,y_test)



r2_score_rf=r2_score(y_pred_rf,y_test)



print('the mean squared error is {}\nr2 score is {}'.format(np.sqrt(mse_rf),r2_score_rf))
# We use random Gradient Boost regression

regressor_gb = GradientBoostingRegressor(random_state=42)



regressor_gb.fit(X_train,y_train)



y_pred_gb=regressor_gb.predict(X_test)
mse_gb=mean_squared_error(y_pred_gb,y_test)



r2_score_gb=r2_score(y_pred_gb,y_test)



print('the mean squared error is {}\nr2 score is {}'.format(np.sqrt(mse_gb),r2_score_gb))
# We use the bagging regressor 



regressor_br=BaggingRegressor(random_state=0)



regressor_br.fit(X_train,y_train)



y_pred_br=regressor_br.predict(X_test)
mse_br=mean_squared_error(y_pred_br,y_test)



r2_score_br=r2_score(y_pred_br,y_test)



print('the mean squared error is {}\nr2 score is {}'.format(np.sqrt(mse_br),r2_score_br))
# We perform the Ada Boost Regressor

regressor_ar=AdaBoostRegressor(random_state=0)



regressor_ar.fit(X_train,y_train)



y_pred_ar=regressor_ar.predict(X_test)
mse_ar=mean_squared_error(y_pred_ar,y_test)



r2_score_ar=r2_score(y_pred_ar,y_test)



print('the mean squared error is {}\nr2 score is {}'.format(np.sqrt(mse_ar),r2_score_ar))
# We perform extreme gradient boosting 



regressor_xgb=XGBRegressor(random_state=0)



regressor_xgb.fit(X_train,y_train)



y_pred_xgb=regressor_xgb.predict(X_test)
mse_xgb=mean_squared_error(y_pred_xgb,y_test)



r2_score_xgb=r2_score(y_pred_xgb,y_test)



print('the mean squared error is {}\nr2 score is {}'.format(np.sqrt(mse_xgb),r2_score_xgb))
from sklearn.linear_model import LinearRegression



regressor_linear=LinearRegression()



regressor_linear.fit(X_train,y_train)



y_pred_lin=regressor_linear.predict(X_test)
mse_lin=mean_squared_error(y_pred_lin,y_test)



r2_score_lin=r2_score(y_pred_lin,y_test)



print('the mean squared error is {}\nr2 score is {}'.format(np.sqrt(mse_lin),r2_score_lin))
y_pred_test=regressor_gb.predict(test)
submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission['SalePrice']=y_pred_test
submission.to_csv('submission.csv',index=False)