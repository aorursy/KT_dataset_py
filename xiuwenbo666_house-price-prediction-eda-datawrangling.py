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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x)) 
import matplotlib.pyplot as plt  # Matlab-style plotting

%matplotlib inline

import seaborn as sns

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from e.g. seaborn)
sns.set_style('darkgrid')
#display the first five rows of the train dataset.

train.head(5)
#display the first five rows of the test dataset.

test.head(5)
print('The training data size before dropping Id feature is : {} '.format(train.shape))

print('The test data size before dropping Id feature is : {} '.format(test.shape))
print('The information of training data is:')      

print(train.info())
print('The information of test data is:')      

print(test.info())
print('The summary of training data is:')      

print(train.describe())
print('The summary of test data is:')      

print(test.describe())
from scipy import stats

from scipy.stats import norm, skew #for some statistics
# Get the fitted parameters used by the function and limit floats output to 2 decimal points.

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
sns.distplot(train['SalePrice'] , fit=norm)

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')
fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
train["SalePrice"] = np.log1p(train["SalePrice"])
# Get the fitted parameters used by the function and plot the distribution 

(mu, sigma) = norm.fit(train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

sns.distplot(train['SalePrice'] , fit=norm)
fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
# check the missing value in training data

missing_train = train.isnull().sum().sort_values(ascending = False)

missing_train
#plot the top 10 missing values

missing_x_axis = missing_train[:10]

missing_y_axis = missing_train[:10].index

width = 10

height = 8

plt.figure(figsize=(width, height))



sns.barplot(missing_x_axis, missing_y_axis)

plt.title('Missing value in trianing data')
# check the top 10 missing values in test data

missing_test = test.isnull().sum().sort_values(ascending = False)



missing_x_axis = missing_test[:10]

missing_y_axis = missing_test[:10].index



width = 10

height = 8

plt.figure(figsize=(width, height))



sns.barplot(missing_x_axis, missing_y_axis)

plt.title('Missing value in test data')
# columns where NaN values have meaning e.g. no pool etc.

cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',

               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',

               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',

               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',

               'MSZoning', 'Utilities']



# replace 'NaN' with 'None' in these columns

for col in cols_fillna:

    train[col].fillna('None',inplace=True)

    test[col].fillna('None',inplace=True)
missing_total = train.isnull().sum().sort_values(ascending=False)

missing_percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([missing_total, missing_percent], axis=1, keys=['Missing Value Total', 'Percent'])

missing_data.head()
# fillna with mean for the remaining columns: LotFrontage, GarageYrBlt, MasVnrArea

cols_fillna = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']



for col in cols_fillna:

    train[col].fillna(train[col].mean(), inplace=True)

    test[col].fillna(test[col].mean(), inplace=True)
missing_total = train.isnull().sum().sort_values(ascending=False)

missing_percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([missing_total, missing_percent], axis=1, keys=['Missing Values Total', 'Percent'])

missing_data.head()
numerical_feats = train.dtypes[train.dtypes != "object"].index

print("Number of Numerical features: ", len(numerical_feats))



categorical_feats = train.dtypes[train.dtypes == "object"].index

print("Number of Categorical features: ", len(categorical_feats))
print(train[numerical_feats].columns)

print("*"*100)

print(train[categorical_feats].columns)
for col in numerical_feats:

    print('{:15}'.format(col), 

          'Skewness: {:05.2f}'.format(train[col].skew()) , 

          '   ' ,

          'Kurtosis: {:06.2f}'.format(train[col].kurt())  

         )
skewed_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF2', 'LowQualFinSF', 'GrLivArea'

                   , 'BsmtHalfBath', 'BsmtFinSF1', 'TotalBsmtSF', 'WoodDeckSF', 'OpenPorchSF'

                   , 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
for feature in skewed_features:

    train[feature] = np.log1p(train[feature])

    test[feature] = np.log1p(test[feature])
len(numerical_feats)
nr_rows = 12

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))



li_num_feats = list(numerical_feats)

li_not_plot = ['Id', 'SalePrice']

li_plot_num_feats = [c for c in list(numerical_feats) if c not in li_not_plot]





for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(li_plot_num_feats):

            sns.regplot(train[li_plot_num_feats[i]], train['SalePrice'], ax = axs[r][c])

            stp = stats.pearsonr(train[li_plot_num_feats[i]], train['SalePrice'])

            #axs[r][c].text(0.4,0.9,"title",fontsize=7)

            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])

            axs[r][c].set_title(str_title,fontsize=11)

            

plt.tight_layout()    

plt.show()   
train = train.drop(

    train[(train['OverallQual']==10) & (train['SalePrice']<12.3)].index)

train = train.drop(

    train[(train['GrLivArea']>8.3) & (train['SalePrice']<12.5)].index)
corr = train.corr()

corr_abs = corr.abs()

min_val_corr = 0.4    





nr_num_cols = len(numerical_feats)

ser_corr = corr_abs.nlargest(nr_num_cols, 'SalePrice')['SalePrice']



cols_abv_corr_limit = list(ser_corr[ser_corr.values > min_val_corr].index)

cols_bel_corr_limit = list(ser_corr[ser_corr.values <= min_val_corr].index)
print(ser_corr)
print("List of numerical features with r above min_val_corr :", cols_abv_corr_limit)
print("List of numerical features with r below min_val_corr :", cols_bel_corr_limit)
for catg in list(categorical_feats) :

    print(train[catg].value_counts())

    print('*'*50)
nr_rows = 15

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*4,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(list(categorical_feats)):

            sns.boxplot(x=list(categorical_feats)[i], y='SalePrice', data=train, ax = axs[r][c])

    

plt.tight_layout()    

plt.show()   
catg_strong_corr = [ 'MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual', 

                     'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']



catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 

                  'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle', 

                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 

                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 

                  'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 

                  'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 

                  'SaleCondition']
def plot_corr_matrix(df, nr_c, targ) :

    

    corr = df.corr()

    corr_abs = corr.abs()

    cols = corr_abs.nlargest(nr_c, targ)[targ].index

    cm = np.corrcoef(df[cols].values.T)



    plt.figure(figsize=(nr_c/1.5, nr_c/1.5))

    sns.set(font_scale=1.25)

    sns.heatmap(cm, linewidths=1.5, annot=True, square=True, 

                fmt='.2f', annot_kws={'size': 10}, 

                yticklabels=cols.values, xticklabels=cols.values

               )

    plt.show()
nr_feats = len(cols_abv_corr_limit)
plot_corr_matrix(train, nr_feats, 'SalePrice')
id_test = test['Id']



to_drop_num  = cols_bel_corr_limit

to_drop_catg = catg_weak_corr



cols_to_drop = ['Id'] + to_drop_num + to_drop_catg 



for df in [train, test]:

    df.drop(cols_to_drop, inplace= True, axis = 1)
catg_list = catg_strong_corr.copy()

catg_list.remove('Neighborhood')



for catg in catg_list :

    sns.boxenplot(x=catg, y='SalePrice', data=train)

    plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(16, 5)

sns.boxenplot(x='Neighborhood', y='SalePrice', data=train, ax=ax)

plt.xticks(rotation=45)

plt.show()
for catg in catg_list :

    group = train.groupby(catg)['SalePrice'].mean()

    print(group)
# 'MSZoning'

msz_catg2 = ['RM', 'RH']

msz_catg3 = ['RL', 'FV'] 





# Neighborhood

nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker']

nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']



# Condition2

cond2_catg2 = ['Norm', 'RRAe']

cond2_catg3 = ['PosA', 'PosN'] 



# SaleType

SlTy_catg1 = ['Oth']

SlTy_catg3 = ['CWD']

SlTy_catg4 = ['New', 'Con']
for df in [train, test]:

    

    df['MSZ_num'] = 1  

    df.loc[(df['MSZoning'].isin(msz_catg2) ), 'MSZ_num'] = 2    

    df.loc[(df['MSZoning'].isin(msz_catg3) ), 'MSZ_num'] = 3        

    

    df['NbHd_num'] = 1       

    df.loc[(df['Neighborhood'].isin(nbhd_catg2) ), 'NbHd_num'] = 2    

    df.loc[(df['Neighborhood'].isin(nbhd_catg3) ), 'NbHd_num'] = 3    



    df['Cond2_num'] = 1       

    df.loc[(df['Condition2'].isin(cond2_catg2) ), 'Cond2_num'] = 2    

    df.loc[(df['Condition2'].isin(cond2_catg3) ), 'Cond2_num'] = 3    

    

    df['Mas_num'] = 1       

    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2 

    

    df['ExtQ_num'] = 1       

    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2     

    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3     

    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4     

   

    df['BsQ_num'] = 1          

    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2     

    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3     

    

    df['CA_num'] = 0          

    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1    



    df['Elc_num'] = 1       

    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2 





    df['KiQ_num'] = 1       

    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2     

    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3     

    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4      

    

    df['SlTy_num'] = 2       

    df.loc[(df['SaleType'].isin(SlTy_catg1) ), 'SlTy_num'] = 1  

    df.loc[(df['SaleType'].isin(SlTy_catg3) ), 'SlTy_num'] = 3  

    df.loc[(df['SaleType'].isin(SlTy_catg4) ), 'SlTy_num'] = 4  
new_col_num = ['MSZ_num', 'NbHd_num', 'Cond2_num'

               , 'Mas_num', 'ExtQ_num', 'BsQ_num'

               , 'CA_num', 'Elc_num', 'KiQ_num', 'SlTy_num']
nr_rows = 4

nr_cols = 3



fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3))



for r in range(0,nr_rows):

    for c in range(0,nr_cols):  

        i = r*nr_cols+c

        if i < len(new_col_num):

            sns.regplot(train[new_col_num[i]], train['SalePrice'], ax = axs[r][c])

            stp = stats.pearsonr(train[new_col_num[i]], train['SalePrice'])

            str_title = "r = " + "{0:.2f}".format(stp[0]) + "      " "p = " + "{0:.2f}".format(stp[1])

            axs[r][c].set_title(str_title,fontsize=11)

            

plt.tight_layout()    
catg_cols_to_drop = ['Neighborhood' , 'Condition2', 'MasVnrType'

                     , 'ExterQual', 'BsmtQual','CentralAir', 'Electrical'

                     , 'KitchenQual', 'SaleType']
corr1 = train.corr()

corr_abs_1 = corr1.abs()



nr_all_cols = len(train)

ser_corr_1 = corr_abs_1.nlargest(nr_all_cols, 'SalePrice')['SalePrice']



print(ser_corr_1)
cols_bel_corr_limit_1 = list(ser_corr_1[ser_corr_1.values <= min_val_corr].index)

for df in [train, test] :

    df.drop(catg_cols_to_drop, inplace= True, axis = 1)

    df.drop(cols_bel_corr_limit_1, inplace= True, axis = 1)    
corr = train.corr()

corr_abs = corr.abs()



nr_all_cols = len(train)

print (corr_abs.nlargest(nr_all_cols, 'SalePrice')['SalePrice'])
nr_feats=len(train.columns)

plot_corr_matrix(train, nr_feats, 'SalePrice')
# switch for dropping columns that are similar to others already used and show a high correlation to these     

drop_similar = 1
cols = corr_abs.nlargest(nr_all_cols, 'SalePrice')['SalePrice'].index

cols = list(cols)



if drop_similar == 1 :

    for col in ['GarageArea','1stFlrSF','TotRmsAbvGrd','GarageYrBlt'] :

        if col in cols: 

            cols.remove(col)
print(list(cols))
feats = cols.copy()

feats.remove('SalePrice')



print(feats)
df_train_ml = train[feats].copy()

df_test_ml  = test[feats].copy()



y = train['SalePrice']
all_data = pd.concat((train[feats], test[feats]))
df_train_ml = all_data[:train.shape[0]]

df_test_ml  = all_data[train.shape[0]:]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

df_train_ml_sc = sc.fit_transform(df_train_ml)

df_test_ml_sc = sc.transform(df_test_ml)
df_train_ml_sc = pd.DataFrame(df_train_ml_sc)

df_test_ml_sc = pd.DataFrame(df_test_ml_sc)
X = df_train_ml.copy()

y = train['SalePrice']

X_test = df_test_ml.copy()



X_sc = df_train_ml_sc.copy()

y_sc = train['SalePrice']

X_test_sc = df_test_ml_sc.copy()
X.info()
X_test.info()