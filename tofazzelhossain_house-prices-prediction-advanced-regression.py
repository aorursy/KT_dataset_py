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

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

import warnings





from scipy import stats

from scipy.stats import skew, norm

from scipy.stats.stats import pearsonr





from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.ensemble import VotingRegressor, GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error





warnings.simplefilter(action='ignore')

sns.set(style = 'white')

%matplotlib inline
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
#Check the data size

print("Data size of train: {}" .format(train.shape))

print("Data size of test: {}" .format(test.shape))
train.head(10)
test.head(10)
# Keep original data clean

train_data = train.copy()

test_data = test.copy()

train_id = train_data['Id'].copy()

test_id = test_data['Id'].copy()
#Check target column's description

train_data['SalePrice'].describe()
#Correlation matrix heatmap

corr_matrix = train_data.corr()

f, ax = plt.subplots(figsize = (10, 10))

sns.heatmap(corr_matrix, vmax = 0.8, square = True)
#Correlation matrix using only numerical features with sale price

num_features = train_data.select_dtypes(exclude='object')

numcorr = num_features.corr()

f, ax = plt.subplots(figsize=(20,2))

sns.heatmap(numcorr.sort_values(by=['SalePrice'], ascending=False).head(1), cmap='Reds')

plt.xticks(weight='bold')

plt.yticks(weight='bold', color='red', rotation=0)

plt.show()
#Check the correlation of top 10 variables

print (numcorr['SalePrice'].sort_values(ascending=False).head(10).to_frame())
# Top 10 variables for heatmap

k = 10

cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

plt.subplots(figsize = (15, 10))

sns.set(font_scale = 1.50)

heatmaps = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, 

                       xticklabels = cols.values, yticklabels = cols.values)

plt.show()
top_corr = pd.DataFrame(cols)

top_corr.columns = ['Top Correlated Features']

top_corr
variable_features = np.array(top_corr['Top Correlated Features'])

variable_features
#check the relations between the top correlated features and SalePrice in box plots

for i in range (1,10):

    sale_price = variable_features[0]

    other_feature = variable_features[i]

    data = pd.concat([train_data[sale_price], train_data[other_feature]], axis = 1)

    f = plt.figure(figsize = (10, 7))

    sns.boxplot(x = train_data[other_feature], y = train_data[sale_price], data = data)

    plt.show()
#check the relations between the top correlated features and SalePrice in joint plots

for i in range (1,10):

    sale_price = variable_features[0]

    other_feature = variable_features[i]

    

    j = sns.jointplot(x = train_data[other_feature], y = train_data[sale_price], kind = 'reg')

    j.annotate(stats.pearsonr)

    plt.show()
#Fixing data

# Removing outliers manually (Two points in the bottom right)

train_data = train_data.drop(train_data[(train_data['GrLivArea']>4000) 

                         & (train_data['SalePrice']<300000)].index).reset_index(drop=True)



# Removing outliers manually (More than 4-cars, less than $300k)

train_data = train_data.drop(train_data[(train_data['GarageCars']>3) 

                         & (train_data['SalePrice']<300000)].index).reset_index(drop=True)



# Removing outliers manually (More than 1000 sqft, less than $300k)

train_data = train_data.drop(train_data[(train_data['GarageArea']>1000) 

                         & (train_data['SalePrice']<300000)].index).reset_index(drop=True)
#combining datasets

ntrain = train_data.shape[0]

ntest = test_data.shape[0]

y_train = train_data.SalePrice.values

all_data = pd.concat((train_data, test_data)).reset_index(drop = True)



#Drop the target "SalePrice" and Id columns

all_data.drop(['Id'], axis = 1, inplace = True)

all_data.drop(['SalePrice'], axis = 1, inplace = True)



print("Train data size is: {}" .format(train_data.shape))

print("Test data size is: {}" .format(test_data.shape))

print("Combined dataset size is: {}" .format(all_data.shape))
#separate categorical and numerical data columns

cat_columns = all_data.select_dtypes(include = ['object']).columns

num_columns = all_data.select_dtypes(include = ['int64', 'float64']).columns
cat = len(cat_columns)

num = len(num_columns)



print('Total features: ', cat, 'categorical', '+', num, 'numerical', '=', cat+num, 'features')
#Find missing ratio of dataset

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending = False)[:25]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data
#Percent missing data by feature

f,ax = plt.subplots(figsize = (10, 8))

plt.xticks(rotation = '90')

sns.barplot(x = all_data_na.index, y = all_data_na)

plt.xlabel('Features', fontsize = 15)

plt.ylabel ('Percent of missing values', fontsize = 15)

plt.title ('Percent missing data by feature', fontsize = 15)



plt.axhline(y=90, color='r', linestyle='-')



plt.text(10, 93.5, 'Columns with more than 90% missing values', fontsize=12, color='crimson', ha='left' ,va='top')

plt.text(10, 87.5, 'Columns with less than 90% missing values', fontsize=12, color='grey', ha='left' ,va='top')
#fill the missing data



for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):

    all_data[col] = all_data[col].fillna('None')

    

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 

            'BsmtHalfBath', 'MasVnrArea'):

    all_data[col] = all_data[col].fillna(0)

    

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType'):

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    

all_data = all_data.drop(['Utilities'], axis = 1) #drop Utilities feature

all_data["Functional"] = all_data["Functional"].fillna("Typ")
#Find missing ratio of dataset

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending = False)[:25]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data
#Convert misinterpreted data

#MSSubClass data change to string type

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)



#Changing OveralCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)



#Year and month sold are transformed into categorical features

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
#separate categorical and numerical data columns

cat_columns = train_data.select_dtypes(include = ['object']).columns

num_columns =train_data.select_dtypes(include = ['int64', 'float64']).columns

cat_train = len(cat_columns)

num_train = len(num_columns)



cat_columns = all_data.select_dtypes(include = ['object']).columns

num_columns = all_data.select_dtypes(include = ['int64', 'float64']).columns

cat_all = len(cat_columns)

num_all = len(num_columns)



print('Total features for train data: ', cat_train, 'categorical', '+', num_train, 'numerical', '=', 

      cat_train+num_train, 'features')

print('Total features for all data: ', cat_all, 'categorical', '+', num_all, 'numerical', '=', 

      cat_all+num_all, 'features')
all_data.head(10)
# Process columns and apply LabelEncoder to categorical features

cols = all_data.select_dtypes(include = ['object']).columns



for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# Check shape        

print('Shape all_data: {}'.format(all_data.shape))
all_data.head()
all_data.describe()
# Adding Total Square Feet feature 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
#Plot histogram of Sale price

(mu, sigma) = norm.fit(train_data['SalePrice'])

print('\u03BC = {:.2f} and \u03C3 = {:.2f}' .format(mu, sigma))



plt.figure(figsize = (10,6))

sns.distplot(train_data['SalePrice'], fit = norm)

plt.ylabel('Frequency')





plot = plt.figure(figsize = (10,6))

stats.probplot(train['SalePrice'], plot = plt)

plt.show()



print("Skewness: %f" % train_data['SalePrice'] .skew())

print("Kurtosis: %f" % train_data['SalePrice'] .kurt())
#Fixing skewed features

train_data["SalePrice"] = np.log1p(train_data["SalePrice"])



##Plot histogram of Sale price

(mu, sigma) = norm.fit(train_data['SalePrice'])

print('\u03BC = {:.2f} and \u03C3 = {:.2f}' .format(mu, sigma))



plt.figure(figsize = (10,6))

sns.distplot(train_data['SalePrice'], fit = norm)

plt.ylabel('Frequency')





plot = plt.figure(figsize = (10,6))

stats.probplot(train['SalePrice'], plot = plt)

plt.show()



y_train = train_data.SalePrice.values



print("Skewness: %f" % train_data['SalePrice'] .skew())

print("Kurtosis: %f" % train_data['SalePrice'] .kurt())
#check the skew of all features

features = all_data.columns



skewed_feature = all_data[features].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)

skewness = pd.DataFrame({'Skewed Features' :skewed_feature})

skewness.head(10)
#fixing skewed features

from scipy.special import boxcox1p

skewness = skewness[abs(skewness) > 0.75]

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    all_data[feat] = boxcox1p(all_data[feat], lam)

    all_data[feat] += 1
skewed_feature = all_data[features].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)

skewness = pd.DataFrame({'Skewed Features' :skewed_feature})

skewness.head()
x_train = all_data[:ntrain]

x_test = all_data[ntrain:]

print(x_train.shape)

print(x_test.shape)
#Cross-validation with k-folds

n_folds = 5



def rmse_cv(model):

    kf = KFold(n_folds, shuffle = True, random_state = 45).get_n_splits(x_train.values)

    rmse = np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring = "neg_mean_squared_error", cv = kf))

    return(rmse)
#various models

lasso = make_pipeline(StandardScaler(), Lasso(alpha = 0.005, random_state = 1))

KRR = make_pipeline(StandardScaler(), KernelRidge(alpha = 1.0, kernel = 'polynomial', degree = 2, coef0 = 1.5))

ENet = make_pipeline(StandardScaler(), ElasticNet(alpha = 0.005, l1_ratio = 0.5, random_state = 2))

GBoost = GradientBoostingRegressor(n_estimators = 3000, learning_rate = 0.05, max_depth = 4, max_features = 'sqrt',

                                  min_samples_leaf = 15, min_samples_split = 10, loss = 'huber', random_state = 5)
#checking performance of various models

score_lasso = rmse_cv(lasso)

print("\nLasso score: {:4f} ({:4f})\n" .format(score_lasso.mean(), score_lasso.std()))



score_KRR = rmse_cv(KRR)

print("Kernel Ridge score: {:4f} ({:4f})\n" .format(score_KRR.mean(), score_KRR.std()))



score_ENet = rmse_cv(ENet)

print("ElasticNet score: {:4f} ({:4f})\n" .format(score_ENet.mean(), score_ENet.std()))



score_GBoost = rmse_cv(GBoost)

print("Gradient Boosting score: {:4f} ({:4f})\n" .format(score_GBoost.mean(), score_GBoost.std()))
#Ensembling previous models in Voting Regressor

vote_reg = VotingRegressor([('Lasso', lasso), ('Elastic', ENet), ('Ridge', KRR), ('GBoost',GBoost)])



score_vote = rmse_cv(vote_reg)

print("\nVote score: {:4f} ({:4f})\n" .format(score_vote.mean(), score_vote.std()))
def rmse(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
#training and prediction the model using Voting Regressor

vote_reg.fit(x_train.values, y_train)



#prediction of training target column using train data

train_pred = vote_reg.predict(x_train.values)

print(rmse(y_train, train_pred))



#predictions using test data

prediction = np.exp(vote_reg.predict(x_test.values))
submission = pd.DataFrame()

submission['Id'] = test_id

submission['SalePrice'] = prediction



submission.to_csv('final_submission.csv',index=False)