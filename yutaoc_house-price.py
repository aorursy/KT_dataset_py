import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import train_test_split

from scipy.stats import norm

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

SalePrice = train['SalePrice']  # Separate the column "SalePrice"

train_len = len(train) # the length og training data

print("The dimensions of training data is: {}".format(train.shape))

print("The dimensions of testing data is: {}".format(test.shape))

train.head()

test.head()

all_data = pd.concat(objs=[train, test], axis=0, sort=False).reset_index(drop=True)  # combine the data

all_data = all_data.fillna(np.nan) # fill the all different kinds of missing data with NaN

all_data.head()

all_data.tail()

all_data.info()

f, ax = plt.subplots(figsize=(12, 9))

g = sns.heatmap(train.corr(),cmap="coolwarm")

g = sns.heatmap(train[["SalePrice", "OverallQual", "GrLivArea", 

                       "TotalBsmtSF", "1stFlrSF", "GarageCars", 

                       "GarageArea", "YearBuilt", "FullBath", 

                       "TotRmsAbvGrd","LotFrontage", "YearRemodAdd", 

                       "MasVnrArea", "BsmtFinSF1","Fireplaces","GarageYrBlt"]].corr(),

                cmap="coolwarm")

g = sns.pairplot(train[["SalePrice", "OverallQual", "GrLivArea", "TotalBsmtSF", 

                        "GarageCars", "YearBuilt", "FullBath", ]], height = 2.5)

train[["SalePrice", "OverallQual", "GrLivArea", "TotalBsmtSF", 

       "GarageCars", "YearBuilt", "FullBath", ]].isnull().sum().sort_values(ascending=False)

selected_feature = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "YearBuilt", "FullBath"]

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_count = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_count.head(20)

all_data = all_data.drop((missing_count[missing_count['Total'] > 1]).index,1)

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0]) # since there is only one missing, we fill it with omst common data

new_data = all_data.select_dtypes(include='object')  # Categorical features

for f in selected_feature:

    new_data[f] = all_data[f]  # Numerical features

new_data.info()

new_data.head()

new_data.tail()
new_data.isnull().sum().sort_values(ascending=False)

new_data['MSZoning'] = new_data['MSZoning'].fillna(new_data['MSZoning'].mode()[0])

new_data['Utilities'] = new_data['Utilities'].fillna(new_data['Utilities'].mode()[0])

new_data['Functional'] = new_data['Functional'].fillna(new_data['Functional'].mode()[0])

new_data['TotalBsmtSF'] = new_data['TotalBsmtSF'].fillna(0)

new_data['Exterior1st'] = new_data['Exterior1st'].fillna(new_data['Exterior1st'].mode()[0])

new_data['Exterior2nd'] = new_data['Exterior2nd'].fillna(new_data['Exterior1st'].mode()[0])

new_data['GarageCars'] = new_data['GarageCars'].fillna(0.0)

new_data['SaleType'] = new_data['SaleType'].fillna(new_data['SaleType'].mode()[0])

new_data['KitchenQual'] = new_data['KitchenQual'].fillna(new_data['KitchenQual'].mode()[0])

new_data.isnull().sum().sort_values(ascending=False)

for col in new_data.dtypes[new_data.dtypes == 'object'].index:

    new_data[col] = new_data[col].astype('category')  # converting to a category dtype

    new_data[col] = new_data[col].cat.codes

print(new_data.shape)

new_data.head()

new_data=(new_data-new_data.mean())/new_data.std()

new_data.head()
SalePrice.describe()

g = sns.distplot(SalePrice, fit=norm)

print('Skewness : {}'.format(SalePrice.skew()))

print('Kurtosis : {}'.format(SalePrice.kurt()))

SalePrice = np.log(SalePrice)

g = sns.distplot(SalePrice, fit=norm);

train_info = new_data[:train_len]

train_label = SalePrice

train = pd.concat([train_info, train_label], axis=1, sort=False)



test_info = new_data[train_len:]

from sklearn.linear_model import ElasticNet, Lasso

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from xgboost import XGBRegressor

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train_info.values, train_label.values, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=42))



ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.8, random_state=3))



KR = KernelRidge(alpha=6, kernel='polynomial', degree=2, coef0=2.5)



GBoost = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05, max_depth=4, 

                                   max_features='sqrt',min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber',random_state =5)



XGBoost = XGBRegressor(n_estimators=2000, learning_rate=0.05, random_state =7)

score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(KR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(XGBoost)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

GBoost.fit(train_info, train_label)

res = GBoost.predict(test_info)

res = np.expm1(res)

print(res)
prediction = pd.DataFrame(res, columns=['SalePrice'])

result = pd.concat([test['Id'], prediction], axis=1)

result.columns
result.to_csv('./submission.csv', index=False)