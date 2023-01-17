#DO NOT DISTURB!

import warnings

warnings.filterwarnings("ignore")



# the King and the Queen of libraries

import pandas as pd

import numpy as np



#friendly stats

from scipy import stats

from scipy.stats import norm, skew



#plots

import matplotlib.pyplot as plt

from matplotlib import rcParams

plt.style.use('ggplot')

import seaborn as sns



#Modeling

from sklearn import ensemble, metrics

from sklearn import linear_model, preprocessing

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.model_selection import ShuffleSplit

from sklearn.kernel_ridge import KernelRidge

import xgboost as xgb
df_train=pd.read_csv('../input/train.csv', index_col='Id')

df_test=pd.read_csv('../input/test.csv', index_col='Id')

#Save the 'Id' column

train_ID = df_train.index

test_ID = df_test.index

print('the Train dataframe shape is:',df_train.shape,' while the Test dataframe shape is:',df_test.shape)
df_train.head(5)
df_test.head(5)
fig, ax = plt.subplots()

ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'], color='blue')

plt.ylabel('Sale_Price', fontsize=13)

plt.xlabel('Ground_Living_Area', fontsize=13)

plt.show()
df_train = df_train.drop((df_train[df_train['GrLivArea']>4000]).index)
#looking at the df_train without the two outliers

fig, ax = plt.subplots()

ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'], color='blue')

plt.ylabel('Sale_Price', fontsize=13)

plt.xlabel('Ground_Living_Area', fontsize=13)

plt.show()
sns.distplot(df_train['SalePrice'] , fit=norm, color='blue');

# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df_train['SalePrice'])

print('mu:',mu,'sigma:',sigma)



#Let's plot the distribution of Sale Price

plt.legend(['Normal dist'],loc='best')

plt.ylabel('Frequency')

plt.title('Sale_Price distribution')
#Numpy Abrakadabra!

SALEPRICE_ABRAKADABRA=df_train["SalePrice"]

SALEPRICE_ABRAKADABRA= np.log1p(SALEPRICE_ABRAKADABRA)



#And now

sns.distplot(SALEPRICE_ABRAKADABRA , fit=norm,color='blue');

# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(SALEPRICE_ABRAKADABRA)

print('mu:',mu,'sigma:',sigma)



#Let's plot again the distribution of Sale Price

plt.legend(['Normal dist'],loc='best')

plt.ylabel('Frequency')

plt.title('Sale_Price distribution')
quantitative = [i for i in df_train.columns if df_train.dtypes[i] != 'object']

quantitative.remove('SalePrice')

qualitative = [j for j in df_train.columns if df_train.dtypes[j] == 'object']

print('Quantitative:',quantitative)

print('')

print('Qualitative:',qualitative)
merged_data = df_train.append(df_test, sort=False).reset_index(drop=True)

print("The size of the merged data is:", merged_data.shape)

ntrain = df_train.shape[0]

ntest = df_test.shape[0]

y_train = df_train.SalePrice.values
missing = merged_data.isnull().sum()

missing = missing[missing > 0]

missing50=missing[missing>=2915/2] #feature with_more than 50 percent of all data missing

missing.sort_values(inplace=True)

del missing['SalePrice']

del missing50['SalePrice']

missing.plot.bar()
print('Features with missing values:\n',missing)

print('')

print('Total of features with missing values:\n',len(missing))

print('')

print('Total of the missing values of the features with more than 50 percent of all data missing:\n',missing50)
merged_data["Alley"] = merged_data["Alley"].fillna("None")



merged_data["Fence"] = merged_data["Fence"].fillna("None")



merged_data["MiscFeature"] = merged_data["MiscFeature"].fillna("None")



merged_data["PoolQC"] = merged_data["PoolQC"].fillna("None")



merged_data["FireplaceQu"] = merged_data["FireplaceQu"].fillna("None")



merged_data["LotFrontage"] = merged_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    merged_data[col] = merged_data[col].fillna('None')

    

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    merged_data[col] = merged_data[col].fillna(0)

    

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    merged_data[col] = merged_data[col].fillna(0)

    

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    merged_data[col] = merged_data[col].fillna('None')

    

merged_data["MasVnrType"] = merged_data["MasVnrType"].fillna("None")



merged_data["MasVnrArea"] = merged_data["MasVnrArea"].fillna(0)



merged_data['MSZoning'] = merged_data['MSZoning'].fillna(merged_data['MSZoning'].mode()[0])



merged_data = merged_data.drop(['Utilities'], axis=1)



merged_data["Functional"] = merged_data["Functional"].fillna("Typ")



merged_data['Electrical'] = merged_data['Electrical'].fillna(merged_data['Electrical'].mode()[0])



merged_data['KitchenQual'] = merged_data['KitchenQual'].fillna(merged_data['KitchenQual'].mode()[0])



merged_data['Exterior1st'] = merged_data['Exterior1st'].fillna(merged_data['Exterior1st'].mode()[0])



merged_data['Exterior2nd'] = merged_data['Exterior2nd'].fillna(merged_data['Exterior2nd'].mode()[0])



merged_data['SaleType'] = merged_data['SaleType'].fillna(merged_data['SaleType'].mode()[0])



merged_data['MSSubClass'] = merged_data['MSSubClass'].fillna("None")
merged_data.head(10)
#Check

merged_data_NA_VALUES =merged_data.isnull().sum()
merged_data['TotalSF'] = merged_data['TotalBsmtSF'] + merged_data['1stFlrSF'] + merged_data['2ndFlrSF']
merged_data['MSSubClass'] = merged_data['MSSubClass'].apply(str)

merged_data['OverallCond'] = merged_data['OverallCond'].astype(str)

merged_data['YrSold'] = merged_data['YrSold'].astype(str)

merged_data['MoSold'] = merged_data['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder

columns = ['LandContour',  'MSZoning',  'Alley',

      'MasVnrType',  'ExterQual',  'ExterCond',

      'BsmtQual',  'BsmtCond',  'BsmtExposure',

      'BsmtFinType1',  'BsmtFinType2', 'HeatingQC',

      'CentralAir',  'KitchenQual',  'FireplaceQu',

      'GarageFinish',  'GarageQual',  'GarageCond',

      'PavedDrive',  'PoolQC',  'MiscFeature']

for col in columns:

    lbl = LabelEncoder() 

    lbl.fit(list(merged_data[col].values)) 

    merged_data[col] = lbl.transform(list(merged_data[col].values))

    

columns_Qual=['LotShape', 'LotConfig', 'LandSlope', 'Neighborhood',

       'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',

       'Foundation', 'Heating', 'Electrical', 'Functional',

       'GarageType', 'GarageCond', 'Fence', 'SaleType',

       'SaleCondition', 'Street']

temp = pd.get_dummies(merged_data[columns_Qual], drop_first=True)

merged_data = merged_data.drop(columns_Qual, axis=1)

merged_data = pd.concat([merged_data, temp], axis=1)
df_train = merged_data[merged_data['SalePrice'].notnull()]

df_test = merged_data[merged_data['SalePrice'].isnull()].drop('SalePrice', axis=1)
x_train = df_train.drop(['SalePrice'], axis=1)

y_train = df_train['SalePrice']

x_test  = df_test
scaler = preprocessing.RobustScaler();

x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)

x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
df_test.shape
KRR = KernelRidge(alpha=0.05, kernel='polynomial', degree=1, coef0=2.5)
lasso = linear_model.Lasso(alpha=0.001, max_iter=5000, random_state=42)
GBoost = ensemble.GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, max_features='sqrt', loss='huber', random_state=42)
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

#It is very useful to create a class in order to make predictions with the above defined models! 



class Averaging_the_models(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, cool_models, peso):

        self.cool_models = cool_models

        self.peso = peso

        

    def fit(self, X, y):

        self.cool_models_ = [clone(x) for x in self.cool_models]

        for model in self.cool_models_:

            model.fit(X, y)

        return self

    

    def predict(self, X):

        predictions = np.column_stack([(model.predict(X) * peso) for model, peso in zip(self.cool_models_, self.peso)])

        return np.sum(predictions, axis=1)
regression = Averaging_the_models(cool_models=(KRR, lasso, GBoost), peso=[0.25, 0.25, 0.50])
regression.fit(x_train_scaled, np.log1p(y_train))

result = np.expm1(regression.predict(x_test_scaled))
subFDS = pd.DataFrame({

    "Id": test_ID,

    "SalePrice": result

})

subFDS.to_csv("subFDS.csv", index=False)
def rmse_cv(model, x, y):

    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=5))

    return rmse



score = rmse_cv(regression, x_train_scaled, np.log1p(y_train))

print(round(score.mean(), 5))
subFDS