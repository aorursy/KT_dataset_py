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
%matplotlib inline 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data.head()
data.tail()
data.shape

data.columns
data.info()
na_values=data.isnull().sum()

na_values=na_values[na_values>0]

na_values.sort_values(inplace=True)

na_values.plot.bar()
drop=["Id","PoolQC","PoolArea","MiscFeature","Alley",'Fence']

data=data.drop(drop,axis=1)
data.describe()
data["SalePrice"].describe()
sns.distplot(data["SalePrice"],kde=True,bins=15)

plt.xticks(rotation=90)
print('Skewness = ',data['SalePrice'].skew())
data["log_SalePrice"] = np.log(data["SalePrice"])

data=data.drop('SalePrice',axis=1)
sns.kdeplot(data["log_SalePrice"])

plt.xticks(rotation=90)
num_features=data.select_dtypes(include=[np.number])

num_features.columns
cat_features=data.select_dtypes(include=[np.object])

cat_features.columns
correlation=num_features.corr()

print(correlation["log_SalePrice"].sort_values(ascending=False))
f,ax=plt.subplots(figsize=(14,12))

plt.title("Correlation of Numeric Features with SalePrice")

sns.heatmap(correlation, vmax=0.8)
k=11

columns=correlation.nlargest(k,"log_SalePrice")["log_SalePrice"].index

print(columns)

cm=np.corrcoef(data[columns].values.T)

f,ax=plt.subplots(figsize=(14,12))

sns.heatmap(cm, vmax=0.8,linewidths=0.01,square=True,annot=True,cmap='viridis',linecolor='white',xticklabels=columns.values,annot_kws={'size':12} , yticklabels=columns.values)
fig,((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9))=plt.subplots(nrows=3,ncols=3,figsize=(14,10))

sns.regplot(x='GarageCars',y="log_SalePrice",data=data,scatter =True,fit_reg=True,ax=ax1)

sns.regplot(x='OverallQual',y="log_SalePrice",data=data,scatter =True,fit_reg=True,ax=ax2)                                                           

sns.regplot(x='GrLivArea',y="log_SalePrice",data=data,scatter =True,fit_reg=True,ax=ax3)

sns.regplot(x='GarageArea',y="log_SalePrice",data=data,scatter =True,fit_reg=True,ax=ax4)

sns.regplot(x='TotalBsmtSF',y="log_SalePrice",data=data,scatter =True,fit_reg=True,ax=ax5)

sns.regplot(x='1stFlrSF',y="log_SalePrice",data=data,scatter =True,fit_reg=True,ax=ax6)                                                          

sns.regplot(x='FullBath',y="log_SalePrice",data=data,scatter =True,fit_reg=True,ax=ax7)                                                           

sns.regplot(x='TotRmsAbvGrd',y="log_SalePrice",data=data,scatter =True,fit_reg=True,ax=ax8)

sns.regplot(x='YearBuilt',y="log_SalePrice",data=data,scatter =True,fit_reg=True,ax=ax9)             
sns.boxplot(data["log_SalePrice"])

plt.xticks(rotation=90)
f,ax=plt.subplots(figsize=(12,8))

fig=sns.boxplot(x='OverallQual',y="log_SalePrice",data= data)

fig.axis(ylim=0,ymax=15)
first_quartile=data["log_SalePrice"].quantile(.25)

third_quartile=data["log_SalePrice"].quantile(.75)

IQR=third_quartile-first_quartile
outlier=third_quartile + 3*IQR
data.drop(data[data["log_SalePrice"]>outlier].index,axis=0,inplace=True)
data.shape
columns_remove=['1stFlrSF','TotalBsmtSF','GarageArea','GarageCars','GarageYrBlt' ,'YearBuilt','TotRmsAbvGrd','GrLivArea', 'BsmtFinSF1'      

,'LotFrontage','WoodDeckSF', '2ndFlrSF',  'OpenPorchSF','HalfBath' , 'LotArea' ,'BsmtFullBath' ,'BsmtUnfSF','BedroomAbvGr' ,'ScreenPorch' ,

                'MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath',  'MiscVal', 'LowQualFinSF',  'YrSold', 'OverallCond','MSSubClass','EnclosedPorch',

                'KitchenAbvGr']
data.drop(columns_remove,axis=1,inplace=True)
data.shape
data.columns
cat_features.shape
for column in cat_features.columns:

     print("\n" + column)

     print(cat_features[column].value_counts())
nacat_values=cat_features.isnull().sum()

nacat_values=nacat_values[nacat_values>0]

nacat_values.sort_values(inplace=True)

nacat_values.plot.bar()
print(cat_features["FireplaceQu"].value_counts())

print(cat_features["GarageCond"].value_counts())

print(cat_features["GarageQual"].value_counts())

print(cat_features["GarageFinish"].value_counts())

print(cat_features["GarageType"].value_counts())

print(cat_features["BsmtFinType2"].value_counts())

print(cat_features["BsmtExposure"].value_counts())

print(cat_features["BsmtFinType1"].value_counts())

print(cat_features["BsmtCond"].value_counts())

print(cat_features["BsmtQual"].value_counts())

print(cat_features["MasVnrType"].value_counts())

print(cat_features["Electrical"].value_counts())

ordinal_features=cat_features[['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',"KitchenQual" ,'FireplaceQu','GarageQual','GarageCond','CentralAir','LotShape','BsmtFinType1','BsmtFinType2','BsmtExposure','PavedDrive']].copy()

bin_map  = {'TA':2,'Gd':3, 'Fa':1,'Ex':4,'Po':1,'None':0,'Y':1,'N':0,'Reg':3,'IR1':2,'IR2':1,'IR3':0,"None" : 0,

            "No" : 2, "Mn" : 2, "Av": 3,"Gd" : 4,"Unf" : 1, "LwQ": 2, "Rec" : 3,"BLQ" : 4, "ALQ" : 5, "GLQ" : 6

            }

ordinal_features['ExterQual'] = ordinal_features['ExterQual'].map(bin_map)

ordinal_features['ExterCond'] = ordinal_features['ExterCond'].map(bin_map)

ordinal_features['BsmtCond'] = ordinal_features['BsmtCond'].map(bin_map)

ordinal_features['BsmtQual'] = ordinal_features['BsmtQual'].map(bin_map)

ordinal_features['HeatingQC'] = ordinal_features['HeatingQC'].map(bin_map)

ordinal_features['KitchenQual'] = ordinal_features['KitchenQual'].map(bin_map)

ordinal_features['FireplaceQu'] = ordinal_features['FireplaceQu'].map(bin_map)

ordinal_features['GarageQual'] = ordinal_features['GarageQual'].map(bin_map)

ordinal_features['GarageCond'] = ordinal_features['GarageCond'].map(bin_map)

ordinal_features['CentralAir'] = ordinal_features['CentralAir'].map(bin_map)

ordinal_features['LotShape'] = ordinal_features['LotShape'].map(bin_map)

ordinal_features['BsmtExposure'] = ordinal_features['BsmtExposure'].map(bin_map)

ordinal_features['BsmtFinType1'] = ordinal_features['BsmtFinType1'].map(bin_map)

ordinal_features['BsmtFinType2'] = ordinal_features['BsmtFinType2'].map(bin_map)



PavedDrive =   {"N" : 0, "P" : 1, "Y" : 2}

ordinal_features['PavedDrive'] = ordinal_features['PavedDrive'].map(PavedDrive)
ordinal_features
ordinal_features = ordinal_features.fillna(0)

ordinal_features
ordinal_features.astype('int')
nominal_list=['MSZoning', 'Street', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType',  'Foundation',

       'Heating',  'Electrical', 'Functional',  'GarageType', 'GarageFinish',

       'SaleType', 'SaleCondition']

#nom_feature=cat_features.drop(['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',"KitchenQual" ,'FireplaceQu','GarageQual','GarageCond'],axis=1,inplace=True)
nominal_features=cat_features[nominal_list]
nominal_features
nominal_features = pd.get_dummies(nominal_features, columns=nominal_features.columns) 

nominal_features
#Using One hot encoder

#from sklearn.preprocessing import OneHotEncoder

# define one hot encoding

#encoder = OneHotEncoder(sparse=False)

# transform data

#onehot = encoder.fit_transform(nominal_features)

#onehot
cat_final = pd.concat([ordinal_features, nominal_features], axis=1,sort=False)

cat_final.head()
data_final=pd.concat([num_features,cat_final],axis=1,sort=False)

data_final