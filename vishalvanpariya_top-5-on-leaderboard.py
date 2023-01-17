#importing libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(font_scale=1)
#importing data from csv file using pandas

train=pd.read_csv('../input/home-data-for-ml-course/train.csv')

test=pd.read_csv('../input/home-data-for-ml-course/test.csv')



train.head()
#lets create scatterplot of GrLivArea and SalePrice

sns.scatterplot(x='GrLivArea',y='SalePrice',data=train)
#as per above plot we can see there are two outliers which can affect on out model,lets remove those outliers

train=train.drop(train.loc[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index,0)

train.reset_index(drop=True, inplace=True)
#lest we how its look after removing outliers

sns.scatterplot(x='GrLivArea',y='SalePrice',data=train)
#lets create heatmap first of all lest see on which feature SalePrice is dependent

corr=train.drop('Id',1).corr().sort_values(by='SalePrice',ascending=False).round(2)

print(corr['SalePrice'])
#here we can see SalePrice mostly dependent on this features OverallQual,GrLivArea,TotalBsmtSF,GarageCars,1stFlrSF,GarageArea 

plt.subplots(figsize=(12, 9))

sns.heatmap(corr, vmax=.8, square=True);
#now lets create heatmap for top 10 correlated features

cols =corr['SalePrice'].head(10).index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1)

hm = sns.heatmap(cm, annot=True, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#lets see relation of 10 feature with SalePrice through Pairplot

sns.pairplot(train[corr['SalePrice'].head(10).index])
#lets store number of test and train rows

trainrow=train.shape[0]

testrow=test.shape[0]
#copying id data

testids=test['Id'].copy()
#copying sales priece

y_train=train['SalePrice'].copy()
#combining train and test data

data=pd.concat((train,test)).reset_index(drop=True)

data=data.drop('SalePrice',1)
#dropping id columns

data=data.drop('Id',axis=1)
#checking missing data

missing=data.isnull().sum().sort_values(ascending=False)

missing=missing.drop(missing[missing==0].index)

missing
#PoolQC is quality of pool but mostly house does not have pool so putting NA

data['PoolQC']=data['PoolQC'].fillna('NA')

data['PoolQC'].unique()
#MiscFeature: mostly house does not have it so putting NA

data['MiscFeature']=data['MiscFeature'].fillna('NA')

data['MiscFeature'].unique()
#Alley,Fence,FireplaceQu: mostly house does not have it so putting NA

data['Alley']=data['Alley'].fillna('NA')

data['Alley'].unique()



data['Fence']=data['Fence'].fillna('NA')

data['Fence'].unique()



data['FireplaceQu']=data['FireplaceQu'].fillna('NA')

data['FireplaceQu'].unique()
#LotFrontage: all house have linear connected feet so putting most mean value

data['LotFrontage']=data['LotFrontage'].fillna(data['LotFrontage'].dropna().mean())
#GarageCond,GarageQual,GarageFinish

data['GarageCond']=data['GarageCond'].fillna('NA')

data['GarageCond'].unique()



data['GarageQual']=data['GarageQual'].fillna('NA')

data['GarageQual'].unique()



data['GarageFinish']=data['GarageFinish'].fillna('NA')

data['GarageFinish'].unique()
#GarageYrBlt,GarageType,GarageArea,GarageCars putting 0

data['GarageYrBlt']=data['GarageYrBlt'].fillna(0)

data['GarageType']=data['GarageType'].fillna(0)

data['GarageArea']=data['GarageArea'].fillna(0)

data['GarageCars']=data['GarageCars'].fillna(0)
#BsmtExposure,BsmtCond,BsmtQual,BsmtFinType2,BsmtFinType1 

data['BsmtExposure']=data['BsmtExposure'].fillna('NA')

data['BsmtCond']=data['BsmtCond'].fillna('NA')

data['BsmtQual']=data['BsmtQual'].fillna('NA')

data['BsmtFinType2']=data['BsmtFinType2'].fillna('NA')

data['BsmtFinType1']=data['BsmtFinType1'].fillna('NA')



#BsmtFinSF1,BsmtFinSF2 

data['BsmtFinSF1']=data['BsmtFinSF1'].fillna(0)

data['BsmtFinSF2']=data['BsmtFinSF2'].fillna(0)
#MasVnrType,MasVnrArea

data['MasVnrType']=data['MasVnrType'].fillna('NA')

data['MasVnrArea']=data['MasVnrArea'].fillna(0)
#MSZoning 

data['MSZoning']=data['MSZoning'].fillna(data['MSZoning'].dropna().sort_values().index[0])
#Utilities

data['Utilities']=data['Utilities'].fillna(data['Utilities'].dropna().sort_values().index[0])
#BsmtFullBath

data['BsmtFullBath']=data['BsmtFullBath'].fillna(0)



#Functional

data['Functional']=data['Functional'].fillna(data['Functional'].dropna().sort_values().index[0])



#BsmtHalfBath

data['BsmtHalfBath']=data['BsmtHalfBath'].fillna(0)



#BsmtUnfSF

data['BsmtUnfSF']=data['BsmtUnfSF'].fillna(0)
#Exterior2nd

data['Exterior2nd']=data['Exterior2nd'].fillna('NA')



#Exterior1st

data['Exterior1st']=data['Exterior1st'].fillna('NA')
#TotalBsmtSF

data['TotalBsmtSF']=data['TotalBsmtSF'].fillna(0)
#SaleType

data['SaleType']=data['SaleType'].fillna(data['SaleType'].dropna().sort_values().index[0])
#Electrical

data['Electrical']=data['Electrical'].fillna(data['Electrical'].dropna().sort_values().index[0])
#KitchenQual

data['KitchenQual']=data['KitchenQual'].fillna(data['KitchenQual'].dropna().sort_values().index[0])
#lets check any missing remain

missing=data.isnull().sum().sort_values(ascending=False)

missing=missing.drop(missing[missing==0].index)

missing
#great no missing data
#as we know some feature are highly co-related with SalePrice so lets create some feature using these features

data['GrLivArea_2']=data['GrLivArea']**2

data['GrLivArea_3']=data['GrLivArea']**3

data['GrLivArea_4']=data['GrLivArea']**4



data['TotalBsmtSF_2']=data['TotalBsmtSF']**2

data['TotalBsmtSF_3']=data['TotalBsmtSF']**3

data['TotalBsmtSF_4']=data['TotalBsmtSF']**4



data['GarageCars_2']=data['GarageCars']**2

data['GarageCars_3']=data['GarageCars']**3

data['GarageCars_4']=data['GarageCars']**4



data['1stFlrSF_2']=data['1stFlrSF']**2

data['1stFlrSF_3']=data['1stFlrSF']**3

data['1stFlrSF_4']=data['1stFlrSF']**4



data['GarageArea_2']=data['GarageArea']**2

data['GarageArea_3']=data['GarageArea']**3

data['GarageArea_4']=data['GarageArea']**4
#lets add 1stFlrSF and 2ndFlrSF and create new feature floorfeet

data['Floorfeet']=data['1stFlrSF']+data['2ndFlrSF']

data=data.drop(['1stFlrSF','2ndFlrSF'],1)
#MSSubClass,MSZoning

data=pd.get_dummies(data=data,columns=['MSSubClass'],prefix='MSSubClass')

data=pd.get_dummies(data=data,columns=['MSZoning'],prefix='MSZoning')

data.head()
#Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle

data=pd.get_dummies(data=data,columns=['Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle'])

data.head()
#OverallQual

data=pd.get_dummies(data=data,columns=['OverallQual'],prefix='OverallQual')
#OverallCond

data=pd.get_dummies(data=data,columns=['OverallCond'],prefix='OverallCond')
#we have remodel year data so lest one new feature home is remodeled or not

data['Remodeled']=0

data.loc[data['YearBuilt']!=data['YearRemodAdd'],'Remodeled']=1

data=data.drop('YearRemodAdd',1)

data=pd.get_dummies(data=data,columns=['Remodeled'])
#creating dummies fo all categorical data

data=pd.get_dummies(data=data,columns=['RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition'])
#lets add all bath in one feature

data['Bath']=data['BsmtFullBath']+data['BsmtHalfBath']*.5+data['FullBath']+data['HalfBath']*.5

data=data.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],1)
#dummies

data=pd.get_dummies(data=data,columns=['BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd'])
#here we  has one more outliers lets replace it with 0

data.loc[data['GarageYrBlt']==2207.,'GarageYrBlt']=0
#great we have done Feature Engineering
#lets import StandardScaler from sklearn for feature scalling

from sklearn.preprocessing import StandardScaler
#lets split data using trainrow data and scale data

x_train=data.iloc[:trainrow]

x_test=data.iloc[trainrow:]

scaler=StandardScaler()

scaler=scaler.fit(x_train)

x_train_scaled=scaler.transform(x_train)

x_test_scaled=scaler.transform(x_test)
#great we have done with feature scalling, now lets do modeling
#we will use all the basic algorithm one by one



#1.LinearRegression

from sklearn.linear_model import LinearRegression

reg_liner=LinearRegression()

reg_liner.fit(x_train_scaled,y_train)

reg_liner.score(x_train_scaled,y_train)
#2.LogisticRegression

from sklearn.linear_model import LogisticRegression

reg_logistic=LogisticRegression()

reg_logistic.fit(x_train_scaled,y_train)

print(reg_logistic.score(x_train_scaled,y_train))
#3.XGBoost one of the powefull ML Algorithm

from xgboost import XGBRegressor



my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

my_model.fit(x_train_scaled, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(x_train_scaled, y_train)], 

             verbose=False)

print(my_model.score(x_train_scaled,y_train))
#4.DecisionTree

from sklearn.tree import DecisionTreeRegressor

tree=DecisionTreeRegressor(criterion='mse',max_depth=3)

tree.fit(x_train_scaled,y_train)

print(tree.score(x_train_scaled,y_train))
#5.Support Vector Regression

from sklearn import svm

svm_model=svm.SVC()

svm_model.fit(x_train_scaled,y_train)

print(svm_model.score(x_train_scaled,y_train))
#6.Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

gnb=GaussianNB()

mnb=MultinomialNB()

gnb.fit(x_train_scaled,y_train)

mnb.fit(x_train,y_train)

print(gnb.score(x_train_scaled,y_train))

print(mnb.score(x_train,y_train))
#7.Random Forest

from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(n_estimators=500)

rfr.fit(x_train_scaled,y_train)

print(rfr.score(x_train_scaled,y_train))