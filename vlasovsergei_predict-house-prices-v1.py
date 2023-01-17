# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

from sklearn.metrics import mean_squared_error

from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox, boxcox_normmax



from bokeh.plotting import figure

from bokeh.io import show, output_notebook

from bokeh.layouts import gridplot

output_notebook()

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/train.csv')

test_df = pd.read_csv('/kaggle/input/test.csv')
train_df.iloc[:5, 60:80]
train_df.describe()
train_df.info()
numeric_features = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 

                    'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',

                    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

                    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

                    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 

                    'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

                    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',  

                    'MoSold', 'YrSold', 'SalePrice']

categorical_features=['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities',

                      'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',

                      'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond',

                      'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating',

                      'HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu',

                      'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence',

                      'MiscFeature','SaleType','SaleCondition']
data = pd.concat((train_df.drop(['SalePrice', 'Id'], axis=1), test_df.drop(['Id'], axis=1)))

data[categorical_features] = data[categorical_features].fillna('None')



data = pd.get_dummies(data, columns=categorical_features)

# test_data.drop(['SalePrice', 'Id'], axis=1, inplace=True)

# data = data.iloc[:len(train_df), :]
def make_hist_plot(column, log=False, box=False):

    if log:

        hist, edges = np.histogram(np.log1p(data[column].dropna()), bins=50)

    elif box:

        hist, edges = np.histogram(boxcox1p(data[column].dropna(), boxcox_normmax(data[column].dropna() + 1)), bins=50)

    else:

        hist, edges = np.histogram(data[column].dropna(), bins=50)

    p = figure(tools='', background_fill_color="#fafafa")

    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],

           fill_color="navy", line_color="white", alpha=0.5)



    p.y_range.start = 0

    p.xaxis.axis_label = 'x'

    p.yaxis.axis_label = 'Pr(x)'

    p.grid.grid_line_color="white"

    return p
data[numeric_features[:-1]] = data[numeric_features[:-1]].fillna(data[numeric_features[:-1]].median())

skewed_feats = data[numeric_features[:-1]].apply(lambda x: skew(x)).sort_values(ascending=False)

skewed_feats = skewed_feats[skewed_feats > .5].index

skewed_feats
column = 'LotArea'

print(data[column].isnull().sum().sum())

counts = data[column].value_counts().iloc[0]

print(counts / len(data))

print(skew(data[column].fillna(data[column].median())))

print(skew(np.log1p(data[column].fillna(data[column].median()))))

print(skew(boxcox1p(data[column].fillna(data[column].median()), boxcox_normmax(data[column].fillna(data[column].median()) + 1))))

p1 = make_hist_plot(column)

p2 = make_hist_plot(column, log=True)

p3 = make_hist_plot(column, box=True)

show(p1)

show(p2)

show(p3)

# log_columns = ['BsmtFinSF2', 'ScreenPorch', 'EnclosedPorch', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', '2ndFlrSF']

# box_columns = ['LotArea', 'LotFrontage', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'BsmtUnfSF', 'TotRmsAbvGrd']

columns_to_drop = ['PoolArea', 'MiscVal', '3SsnPorch', 'LowQualFinSF', 'KitchenAbvGr']
for col in skewed_feats:

    data[col] = np.log1p(data[col])



data = data.drop(columns_to_drop, axis=1)



# for col in log_columns:

#     data[col] = np.log1p(data[col])



# for col in box_columns:

#     data[col] = boxcox1p(data[col], boxcox_normmax(data[col] + 1))
data.describe()
x, y = data.iloc[:len(train_df), :], np.log1p(train_df['SalePrice'])
model = xgb.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=3,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42)
model.fit(x, y)
xgb.plot_importance(model)

plt.show()
test_data = data.iloc[len(train_df):, :]

test_df['Prediction'] = np.floor(np.expm1(model.predict(test_data)))

filename = 'submission.csv'

pd.DataFrame({'Id': test_df.Id, 'SalePrice': test_df.Prediction}).to_csv(filename, index=False)
print(test_df['Prediction'].head())

print(test_df['Prediction'].count())
# most_relevant_features = ['LotArea', 'BsmtUnfSF', 'GrLivArea', '1stFlrSF', 

#                           'GarageArea', 'TotalBsmtSF', 'LotFrontage', 'GarageYrBlt',

#                           'OpenPorchSF', 'WoodDeckSF', 'BsmtFinSF1', 'YearBuilt', 

#                           'MoSold', 'YearRemodAdd', '2ndFlrSF', 'MasVnrArea']

# data.columns[:30]
# plt.figure(figsize=(6,6))

# plt.scatter(x['GarageArea'], y, c = "blue", marker = "s")

# plt.title("YearBuilt vs SalePrice")

# plt.xlabel("YearBuilt")

# plt.ylabel("SalePrice")

# plt.show()
# data_set = pd.concat((x, y), axis=1)
# data_set = data_set[data_set.SalePrice<13]

# data_set = data_set[data_set.SalePrice>10.75]



# data_set = data_set[data_set.LotFrontage<5.2]

# data_set = data_set[data_set.LotFrontage>3.3]



# data_set = data_set[data_set.LotArea<11.5]

# data_set = data_set[data_set.LotArea>7.5]



# data_set = data_set[data_set['1stFlrSF']<8.2]

# data_set = data_set[data_set['1stFlrSF']>6.]



# data_set = data_set[data_set['GrLivArea']<8.2]

# data_set = data_set[data_set['GrLivArea']>6.2]





# data_set = data_set[data_set.MasVnrArea>2]

# data_set = data_set[data_set.MasVnrArea>2]
# x, y = data_set, data_set['SalePrice']
# model.fit(x, y)
# test_df['Prediction'] = np.floor(np.expm1(model.predict(test_df)))

# filename = 'submission.csv'

# pd.DataFrame({'Id': test_df.Id, 'SalePrice': test_df.Prediction}).to_csv(filename, index=False)
# print(test_df['Prediction'].head())

# print(test_df['Prediction'].count())