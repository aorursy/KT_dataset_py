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
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train.info()
train['SalePrice'].describe()
import seaborn as sns

import matplotlib.pyplot as plt

sns.distplot(train['SalePrice'])

plt.xticks(rotation = 30)
print("Skewness = ", train['SalePrice'].skew())
# correlation

corr = train.corr()



plt.figure(figsize=(15,12))



sns.heatmap(corr,vmax=0.9,square=True)

plt.show();
plt.scatter(x=train['TotRmsAbvGrd'], y=train['GrLivArea'])

plt.xlabel('TotRmsAbvGrd')

plt.ylabel('GrLivArea')

plt.show();
# GarageYrBlt and YearBuilt

plt.scatter(x=train['GarageYrBlt'], y=train['YearBuilt'])

plt.xlabel('GarageYrBlt')

plt.ylabel('YearBuilt')

plt.show();
# 1stFlrSF and TotalBsmtSF

plt.scatter(x=train['1stFlrSF'], y=train['TotalBsmtSF'])

plt.xlabel('1stFlrSF')

plt.ylabel('TotalBsmtSF')

plt.show();
# GarageCars and SalePrice

plt.scatter(x=train['GarageCars'], y=train['GarageArea'])

plt.xlabel('GarageCars')

plt.ylabel('GarageArea')

plt.show();
# correlation

corr = train.corr()

# sort in descending order

corr_top = corr['SalePrice'].sort_values(ascending=False)[:10]

top_features = corr_top.index[1:]



corr_top


# Top features and SalePrice

numeric_cols = ['GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','YearBuilt']

nominal_cols = ['OverallQual','GarageCars','FullBath','TotRmsAbvGrd']



fig,ax=plt.subplots(nrows=9,ncols=1,figsize=(6,30))

for i in range(len(top_features)):    



    ax[i].scatter(x=train[top_features[i]], y=train['SalePrice'])

    ax[i].set_xlabel('%s'%(top_features[i]))

    ax[i].set_ylabel('SalePrice')



plt.tight_layout()

plt.savefig('./Top_featuresvsSalePrice.jpg',dpi=300,bbox_inches='tight')

plt.show();
Q1 = []

Q3 = []

Lower_bound = []

Upper_bound = []

Outliers = []





for i in top_features:

    

    # 25th and 75th percentiles

    q1, q3 = np.percentile(train[i],25), np.percentile(train[i],75)

    # Interquartile range

    iqr = q3 - q1

    # Outlier cutoff

    cut_off = 1.5*iqr

    # Lower and Upper bounds

    lower_bound = q1-cut_off

    upper_bound = q3+cut_off

        

    # save outlier indexes

    outlier = [x for x in train.index if train.loc[x,i]<lower_bound or train.loc[x,i]>upper_bound]

    

    # append values for DataFrame

    Q1.append(q1)

    Q3.append(q3)

    Lower_bound.append(lower_bound)

    Upper_bound.append(upper_bound)

    Outliers.append(len(outlier))

    

    try:

        train.drop(outlier,inplace=True,axis=0)

    except:

        continue



df_out = pd.DataFrame({'Column':top_features,'Q1':Q1,'Q3':Q3,'Lower bound':Lower_bound,'Upper_bound':Upper_bound,'No. of outliers':Outliers})    

df_out.sort_values(by='No. of outliers',ascending=False)
ntrain = train.shape[0]



target = np.log(train["SalePrice"])



train.drop(["Id","SalePrice"],inplace = True, axis=1)



test_id = test['Id']



test.drop('Id',inplace = True,axis =1)



train = pd.concat([train,test])
train.isna().sum().sort_values(ascending=False).head(10)
train['PoolQC'].unique()
#                                                Ordinal features

#NA means no Pool

train['PoolQC'].replace(['Ex','Gd','TA','Fa',np.nan],[4,3,2,1,0],inplace=True)



# NA means no fence

train['Fence'].replace(['GdPrv','MnPrv','GdWo','MnWw',np.nan],[4,3,2,1,0],inplace=True)



# NA means no fireplace

train['FireplaceQu'].replace(['Ex','Gd','TA','Fa','Po',np.nan],[5,4,3,2,1,0],inplace=True)



#                                                 Nominal features

# NA means no miscellaneous feature

train['MiscFeature'].fillna('None',inplace=True)



# NA means no alley access

train['Alley'].fillna('None',inplace=True)



#                                               Numerical features

# Replace null lotfrontage with average of the neighborhood

train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))

# garagefeatures = ['GarageCond','GarageQual','GarageYrBlt','GarageFinish','GarageType'] 



# for i in garagefeatures:

#     print(i,train[i].unique())

    

train['GarageCars'].unique()

#train['GarageArea'].unique()
train['GarageYrBlt'].median()
for i in ['GarageCond','GarageQual']:

    train[i].replace(['Ex','Gd','TA','Fa','Po',np.nan],[5,4,3,2,1,0],inplace=True)

    

train['GarageFinish'].replace(['Fin','RFn','Unf',np.nan],[3,2,1,0],inplace=True)



train['GarageType'].fillna('None',inplace=True)



train['GarageYrBlt'].fillna(train['GarageYrBlt'].median(),inplace = True)

train['GarageArea'].fillna(train['GarageYrBlt'].median(),inplace = True)

train['GarageCars'].fillna(0,inplace=True)

#                                                Ordinal features

for i in ['BsmtCond','BsmtQual']:

    train[i].replace(['Ex','Gd','TA','Fa','Po',np.nan],[5,4,3,2,1,0],inplace=True)



train['BsmtExposure'].replace(['Gd','Av','Mn','No',np.nan],[4,3,2,1,0],inplace=True)



for i in ['BsmtFinType1','BsmtFinType2']:

    train[i].replace(['GLQ','ALQ','BLQ','Rec','LwQ','Unf',np.nan],[6,5,4,3,2,1,0],inplace=True)     



#                                               Numerical features

for i in ['BsmtHalfBath','BsmtFullBath','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']:

    train[i].fillna(0,inplace=True)




















#NA means no masonary work

train['MasVnrType'].fillna('None',inplace=True)



#If no masonary work, then area is 0

train['MasVnrArea'].fillna(0,inplace=True)



#Replace with the most common value

for i in ['MSZoning','Utilities']:

    train[i].fillna(train[i].mode()[0],inplace=True)



#"Assume typical unless deductions are warranted"

train['Functional'].fillna('Typ',inplace=True)



#Replace with others

train['SaleType'].fillna('Oth',inplace=True)

#Replace with most common value

train['Electrical'].fillna(train['Electrical'].mode()[0],inplace=True)



#Replace with 'Other' value

for i in ['Exterior1st','Exterior2nd']:

    train[i].fillna('Other',inplace=True)

    

#Replace with most common value

train['KitchenQual'].fillna(train['KitchenQual'].mode()[0],inplace=True)

#ordinal value

train['KitchenQual'].replace(['Ex','Gd','TA','Fa','Po'],[4,3,2,1,0],inplace=True)
#                                                Ordinal features

train['CentralAir'].replace(['N','Y'],[0,1],inplace=True)

#                                                 Nominal features

for i in ['HeatingQC','ExterCond','ExterQual']:

    train[i].replace(['Ex','Gd','TA','Fa','Po'],[4,3,2,1,0],inplace=True)
# Total surface area of house

train['TotalSF'] = train.apply(lambda x: x['1stFlrSF'] + x['2ndFlrSF'] + x['TotalBsmtSF'], axis=1)



# Total number of bathrooms in the house

train['TotalBath'] = train.apply(lambda x: x['FullBath'] + 0.5*x['HalfBath'] + x['BsmtFullBath'] + 0.5*x['BsmtHalfBath'], axis=1)



# Total Porch area in the house

train['TotalPorch'] = train.apply(lambda x: x['OpenPorchSF'] + x['EnclosedPorch'] + x['3SsnPorch'] + x['ScreenPorch'], axis=1)



# New house or an old house

train['NewHouse'] = train.apply(lambda x: 1 if x['SaleCondition']=='Partial' else 0, axis=1)
# One-Hot encoding

train = pd.get_dummies(train,drop_first=True)

train.head()
# train dataset

df = train.iloc[:ntrain,:]



# test dataset

test = train.iloc[ntrain:,:]
from sklearn.model_selection import train_test_split



X = df

y = target



# training and validation set

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=27)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



lr = LinearRegression()



lr.fit(X_train,y_train)



rmse = np.sqrt(mean_squared_error(y_test,lr.predict(X_test)))

print(rmse)
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error



# different alpha values

alphas = [0.01, 0.1, 0.3, 1, 3, 5, 10, 15, 20,25,28,29,30,31,32,33,34,35]



for a in alphas:



    lr = Ridge(alpha=a)

    

    lr.fit(X_train,y_train)

    

    rmse = np.sqrt(mean_squared_error(y_test,lr.predict(X_test)))

    print('For Alpha = ',a,', RMSE = ',rmse)
model = Ridge(alpha=31)

model.fit(X_train,y_train)
log_pred = model.predict(test)

actual_pred = np.exp(log_pred)
data_dict = {'Id':test_id,'SalePrice':actual_pred}



submit = pd.DataFrame(data_dict)

submit.to_csv('submission.csv',index=False)