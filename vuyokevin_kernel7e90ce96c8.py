import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler

from scipy import stats

from scipy.stats import skew

from sklearn.linear_model import ElasticNet, Ridge



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

from IPython.display import display, HTML

import warnings

warnings.filterwarnings('ignore')
display(HTML("""

<style>

.output_png {

    display: table-cell;

    text-align: center;

    vertical-align: middle;

    }

"""))
#reading the train and test data

test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')
#plotting the salesprice distribution

ax1 = sns.distplot(train['SalePrice'])

ax1.set_title('Distplot Normal Distribution')



#Creating QQ - plot

fig = plt.figure()

sx1 = stats.probplot(train['SalePrice'], plot=plt)

plt.show()



#applying np.log to distribution

train['SalePrice'] =np.log1p(train['SalePrice'])



#New salesprice distribution(After np.log)

ax2 = sns.distplot(train['SalePrice'])

ax2.set_title('Distplot after log transformation')



#New QQ plot Distribution(After np.log)

fig = plt.figure()

sx2 = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
#Creating y-train data

y_train = train.SalePrice.values
#combining train and test data

df = pd.concat([train,test], sort=False)
#Getting datatypes for each category

df.dtypes



#classifying numerical and categorical features

numFeats = df.dtypes[df.dtypes != "object"].index

catFeats = df.dtypes[df.dtypes == "object"].index

print(numFeats)

print(catFeats)
#log transform skewed numeric features:

skewed_feats = train[numFeats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



df[skewed_feats] = np.log1p(df[skewed_feats])
#Counts percentage of null values for each column

for x in df:

    if round(df[x].isna().mean() * 100, 2) > 0:

        print(x,  round(df[x].isna().mean() * 100, 2),'%' )
#Numerical value imputation

for x in df[numFeats]:

    df[x].fillna((round(df[x].mean(),0)), inplace=True)

    

#MANUAL CATEGORICAL IMPUTATION

#Most Frequent Value Imputation

df[ 'BsmtCond' ].fillna('TA' , inplace = True)

df[ 'BsmtExposure' ].fillna( 'No', inplace = True)

df[ 'Electrical' ].fillna('SBrkr' , inplace = True)

df[ 'Exterior1st' ].fillna('VinylSd' , inplace = True)

df[ 'Exterior2nd' ].fillna('VinylSd' , inplace = True)

df[ 'ExterCond' ].fillna('TA' , inplace = True)

df[ 'ExterQual' ].fillna('TA' , inplace = True)

df[ 'Functional' ].fillna('Typ' , inplace = True)

df[ 'KitchenQual' ].fillna('TA' , inplace = True)

df['MSZoning'].fillna('RL', inplace = True)

df[ 'SaleType' ].fillna( 'WD', inplace = True)





#####New Category 'None' Imputation

df[ 'Alley' ].fillna('None' , inplace = True)

df[ 'BsmtFinType1' ].fillna( 'None', inplace = True)

df[ 'BsmtFinType2' ].fillna( 'None', inplace = True)

df[ 'BsmtQual' ].fillna('None' , inplace = True)

df[ 'Fence' ].fillna('None' , inplace = True)

df[ 'FireplaceQu' ].fillna('None' , inplace = True)

df[ 'Foundation' ].fillna('None' , inplace = True)

df[ 'GarageCond' ].fillna( 'None', inplace = True)

df[ 'GarageFinish' ].fillna( 'None', inplace = True)

df[ 'GarageQual' ].fillna( 'None', inplace = True)

df[ 'GarageType' ].fillna('None' , inplace = True)

df[ 'MasVnrType' ].fillna('None' , inplace = True)

df[ 'MiscFeature' ].fillna( 'None', inplace = True)

df[ 'PoolQC' ].fillna('None' , inplace = True)
# Heatmap for correlations between variables

corrmat = df.corr()

f, ax = plt.subplots(figsize=(20, 9))

sns.heatmap(corrmat, vmax=.8, square=True);



# Heatmap for most correlated variables

corrmat = df.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.35]

plt.figure(figsize=(10,10))

g = sns.heatmap(df[top_corr_features].corr(),annot=True)
# Box plot for categorical variables

li_cat_feats = ['Alley','LotShape','Exterior1st','Exterior2nd','BsmtFinType1', 'BsmtFinType2', 'LandContour', 'Neighborhood', 'Condition1', 'Condition2',

      'HouseStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'Heating',

      'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual','FireplaceQu', 'GarageType',

      'GarageFinish', 'GarageQual','PavedDrive','SaleType', 'SaleCondition']

target = 'SalePrice'

nr_rows = 7

nr_cols = 4

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))

for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_cat_feats):

            sns.boxplot(x=li_cat_feats[i], y=target, data=df, ax = axs[r][c])

plt.tight_layout()    

plt.show()
# Scatter Plot for numerical variables

li_num_feats = ['3SsnPorch','LotFrontage', 'BedroomAbvGr','LotArea', 'OverallQual','YearBuilt', 'MasVnrArea', 'BsmtFinSF1','BsmtFinSF2',

        'TotalBsmtSF','BsmtFullBath','BsmtHalfBath','BsmtUnfSF','GrLivArea', 'FullBath','Fireplaces', 'GarageArea', 'WoodDeckSF','OpenPorchSF', 'OverallCond','TotRmsAbvGrd','WoodDeckSF', 'PoolArea']   

target = 'SalePrice'

nr_rows = 6

nr_cols = 4

fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))

for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_num_feats):

            sns.regplot(x=li_num_feats[i], y=target, data=df,  ax = axs[r][c])

    

plt.tight_layout()    

plt.show()
X = df[li_num_feats+li_cat_feats]
X = pd.get_dummies(X, drop_first =True)
#Split train and test data

X_train = X.iloc[:1460, :]

X_test = X.iloc[1460:, :]
sc=RobustScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)
ridge = Ridge()

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

ridge_pred = np.expm1(ridge_pred)
ENet =ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)

ENet.fit(X_train, y_train)

ENet_pred = ENet.predict(X_test)

ENet_pred = np.expm1(ENet_pred)
final_model = ((ridge_pred)*0.30 + (ENet_pred)*0.7)