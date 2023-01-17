# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline

from scipy.stats import skew



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import linear_model

from sklearn import cross_validation

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')



train_df = train_df[['Id', 'LotArea', 'SalePrice', 'TotalBsmtSF', 'GrLivArea','GarageCars', 'YearBuilt', 'YrSold', 'PoolArea', 'TotRmsAbvGrd', 'FullBath', 'HalfBath', 'KitchenAbvGr', 'BedroomAbvGr', 'GarageArea', 'BsmtFullBath','BsmtHalfBath', 'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','LotFrontage','BsmtUnfSF','LowQualFinSF','BsmtFinSF1','BsmtFinSF2', 'MSZoning', 'BldgType', 'HouseStyle', 'Neighborhood', 'Condition1', 'Condition2', 'OverallQual', 'OverallCond', 'Street', 'LotConfig', 'ExterCond', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'GarageCond', 'SaleType', 'SaleCondition']]

#train_df['TotalBuildArea'] = train_df['GrLivArea'] + train_df['TotalBsmtSF']

#train_df['TotalBathAbvGr'] = train_df['FullBath'] + train_df['HalfBath']

train_df['NumOfYear'] = train_df['YrSold'] - train_df['YearBuilt']

#train_df['avgPrice'] = train_df['SalePrice'] / train_df['GrLivArea']

train_df.drop(['Id', 'YrSold', 'YearBuilt'], axis=1, inplace=True)

#train_df['TotalBathBsmt'] = train_df['BsmtFullBath'] + train_df['BsmtHalfBath']



train_df = train_df.loc[train_df['GrLivArea']<3500]

train_df = train_df.loc[train_df['GrLivArea']>500]

#train_df = train_df.loc[train_df['SalePrice']<400000]

#train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

train_df = train_df.loc[train_df['LotArea']<20000]

train_df['LotFrontage'] = train_df['LotFrontage'].fillna(0)

train_df['BsmtCond'] = train_df['BsmtCond'].fillna('TA')

train_df['GarageCond'] = train_df['GarageCond'].fillna('TA')

train_df.info()
# Differentiate numerical features (minus the target) and categorical features

categorical_features = train_df.select_dtypes(include = ["object"]).columns

numerical_features = train_df.select_dtypes(exclude = ["object"]).columns

#numerical_features = numerical_features.drop("SalePrice")

print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))

train_num = train_df[numerical_features]

train_cat = train_df[categorical_features]
skewness = train_num.apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.75]

print(str(skewness.shape[0]) + " skewed numerical features to log transform")

skewed_features = skewness.index

train_df[skewed_features] = np.log1p(train_df[skewed_features])
train_df[skewed_features] = np.log1p(train_df[skewed_features])
train_df = pd.concat([train_num, train_cat], axis = 1)
mszoning_dummies_train = pd.get_dummies(train_df['MSZoning'])

bldgType_dummies_train = pd.get_dummies(train_df['BldgType'])

houseStyle_dummies_train = pd.get_dummies(train_df['HouseStyle'])

houseStyle_dummies_train.drop(['2.5Fin'], axis=1, inplace=True)



neigh_dummies_train = pd.get_dummies(train_df['Neighborhood'])

condition1_dummies_train = pd.get_dummies(train_df['Condition1'])

condition2_dummies_train = pd.get_dummies(train_df['Condition2'])

condition2_dummies_train['RRNe'] = 0

condition_dummies_train = condition1_dummies_train | condition2_dummies_train



street_dummies_train = pd.get_dummies(train_df['Street'])

lotConfig_dummies_train = pd.get_dummies(train_df['LotConfig'])

exterCond_dummies_train = pd.get_dummies(train_df['ExterCond'])

saleType_dummies_train = pd.get_dummies(train_df['SaleType'])

saleCond_dummies_train = pd.get_dummies(train_df['SaleCondition'])



bsmtCond_dummies_train = pd.get_dummies(train_df['BsmtCond'])

bsmtCond_dummies_train.columns = ['BC_Fa', 'BC_Gd', 'BC_Po', 'BC_TA']



heatingQC_dummies_train = pd.get_dummies(train_df['HeatingQC'])

heatingQC_dummies_train.columns = ['HQC_Ex', 'HQC_Fa', 'HQC_Gd', 'HQC_Po', 'HQC_TA']



kitchen_dummies_train = pd.get_dummies(train_df['KitchenQual'])

kitchen_dummies_train.columns = ['KQC_Ex', 'KQC_Fa', 'KQC_Gd', 'KQC_TA']



garageCond_dummies_train = pd.get_dummies(train_df['GarageCond'])

garageCond_dummies_train.columns = ['GC_Ex', 'GC_Fa', 'GC_Gd', 'GC_Po', 'GC_TA']



#mszoning_dummies_train.describe()

#mszoning_dummies_train.drop(['MSZoning'], axis=1, inplace=True)



train_df = train_df.join(mszoning_dummies_train)

train_df.drop(['MSZoning'], axis=1, inplace=True)



train_df = train_df.join(bldgType_dummies_train)

train_df.drop(['BldgType'], axis=1, inplace=True)



train_df = train_df.join(houseStyle_dummies_train)

train_df.drop(['HouseStyle'], axis=1, inplace=True)



train_df = train_df.join(neigh_dummies_train)

train_df.drop(['Neighborhood'], axis=1, inplace=True)



train_df = train_df.join(condition_dummies_train)

train_df.drop(['Condition1'], axis=1, inplace=True)

train_df.drop(['Condition2'], axis=1, inplace=True)



train_df = train_df.join(street_dummies_train)

train_df.drop(['Street'], axis=1, inplace=True)



train_df = train_df.join(lotConfig_dummies_train)

train_df.drop(['LotConfig'], axis=1, inplace=True)



train_df = train_df.join(exterCond_dummies_train)

train_df.drop(['ExterCond'], axis=1, inplace=True)



train_df = train_df.join(saleType_dummies_train)

train_df.drop(['SaleType'], axis=1, inplace=True)



train_df = train_df.join(saleCond_dummies_train)

train_df.drop(['SaleCondition'], axis=1, inplace=True)



train_df = train_df.join(bsmtCond_dummies_train)

train_df.drop(['BsmtCond'], axis=1, inplace=True)



train_df = train_df.join(heatingQC_dummies_train)

train_df.drop(['HeatingQC'], axis=1, inplace=True)



train_df = train_df.join(kitchen_dummies_train)

train_df.drop(['KitchenQual'], axis=1, inplace=True)



train_df = train_df.join(garageCond_dummies_train)

train_df.drop(['GarageCond'], axis=1, inplace=True)

#zoning_train_df = train_df['MSZoning']

#train_df[['MSZoning', 'Id']].groupby(['MSZoning'],as_index=False).count()



#train_df['MSZoning'].loc[train_df['MSZoning'] == "C (all)"] = 'C'

#train_df[['MSZoning', 'Id']].groupby(['MSZoning'],as_index=False).count()



#fig, (axis1) = plt.subplots(1,1,figsize=(10,5))

#sns.countplot(x='YearBuilt', data=train_df, ax=axis1)
test_df = pd.read_csv('../input/test.csv')

test_df = test_df[['Id', 'LotArea', 'TotalBsmtSF', 'GrLivArea','GarageCars', 'YearBuilt', 'YrSold', 'PoolArea', 'TotRmsAbvGrd', 'FullBath', 'HalfBath', 'KitchenAbvGr', 'BedroomAbvGr', 'GarageArea','BsmtFullBath','BsmtHalfBath', 'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','LotFrontage','BsmtUnfSF','LowQualFinSF','BsmtFinSF1','BsmtFinSF2', 'MSZoning', 'BldgType', 'HouseStyle', 'Neighborhood', 'Condition1', 'Condition2', 'OverallQual', 'OverallCond', 'Street', 'LotConfig', 'ExterCond', 'BsmtCond', 'HeatingQC', 'KitchenQual','GarageCond', 'SaleType', 'SaleCondition']]



#test_df['TotalBathAbvGr'] = test_df['FullBath'] + test_df['HalfBath']

test_df['NumOfYear'] = test_df['YrSold'] - test_df['YearBuilt']

test_df['NumOfYear'].loc[test_df['NumOfYear'] < 0] = 0

test_df_Id = test_df['Id']

test_df.drop(['Id', 'YrSold', 'YearBuilt'], axis=1, inplace=True)



#LotArea

avg_lotArea_test = test_df['LotArea'].mean()

std_lotArea_test = test_df['LotArea'].std()

count_outliner_lotArea_test = test_df['LotArea'].loc[test_df['LotArea']>20000].count()



rand_1 = np.random.randint(avg_lotArea_test - std_lotArea_test, avg_lotArea_test + std_lotArea_test, size = count_outliner_lotArea_test)

test_df["LotArea"][test_df['LotArea']>20000] = rand_1



avg_liveArea_test = test_df['GrLivArea'].mean()

test_df['GrLivArea'].loc[test_df['GrLivArea'] > 3500] = avg_liveArea_test





mean_BsmtSF = test_df['TotalBsmtSF'].mean()

mean_GarageCars = test_df['GarageCars'].mean()

mean_GarageArea = test_df['GarageArea'].mean()

mean_BsmtFullBath = test_df['BsmtFullBath'].mean()

mean_BsmtHalfBath = test_df['BsmtHalfBath'].mean()

mean_BsmtUnfSF = test_df['BsmtUnfSF'].mean()

mean_BsmtFinSF1 = test_df['BsmtFinSF1'].mean()

mean_BsmtFinSF2 = test_df['BsmtFinSF2'].mean()



test_df['TotalBsmtSF'] = test_df['TotalBsmtSF'].fillna(mean_BsmtSF)

test_df['GarageCars'] = test_df['GarageCars'].fillna(mean_GarageCars)

test_df['GarageArea'] = test_df['GarageArea'].fillna(mean_BsmtSF)

test_df['BsmtFullBath'] = test_df['BsmtFullBath'].fillna(mean_GarageCars)

test_df['BsmtHalfBath'] = test_df['BsmtHalfBath'].fillna(mean_BsmtSF)

test_df['BsmtUnfSF'] = test_df['BsmtUnfSF'].fillna(mean_GarageCars)

test_df['BsmtFinSF1'] = test_df['BsmtFinSF1'].fillna(mean_BsmtSF)

test_df['BsmtFinSF2'] = test_df['BsmtFinSF2'].fillna(mean_BsmtSF)

test_df['MSZoning'] = test_df['MSZoning'].fillna('RL')

test_df['LotFrontage'] = test_df['LotFrontage'].fillna(0)

#test_df['BsmtCond'] = test_df['BsmtCond'].fillna('TA')

#test_df['KitchenQual'] = test_df['KitchenQual'].fillna('TA')

#test_df['GarageCond'] = test_df['GarageCond'].fillna('TA')

#test_df['SaleType'] = test_df['SaleType'].fillna('WD')



test_df.info()
# Differentiate numerical features (minus the target) and categorical features

test_categorical_features = test_df.select_dtypes(include = ["object"]).columns

test_numerical_features = test_df.select_dtypes(exclude = ["object"]).columns

#numerical_features = numerical_features.drop("SalePrice")

print("Numerical features : " + str(len(test_numerical_features)))

print("Categorical features : " + str(len(test_categorical_features)))

test_num = test_df[test_numerical_features]

test_cat = test_df[test_categorical_features]
skewness = test_num.apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.5]

print(str(skewness.shape[0]) + " skewed numerical features to log transform")

skewed_features = skewness.index

test_df[skewed_features] = np.log1p(test_df[skewed_features])

test_df = pd.concat([test_num, test_cat], axis = 1)
mszoning_dummies_test = pd.get_dummies(test_df['MSZoning'])

#mszoning_dummies_test.drop(['MSZoning'], axis=1, inplace=True)

bldgType_dummies_test = pd.get_dummies(test_df['BldgType'])

houseStyle_dummies_test = pd.get_dummies(test_df['HouseStyle'])

#houseStyle_dummies_test['2.5Fin'] = 0

neigh_dummies_test = pd.get_dummies(test_df['Neighborhood'])

condition1_dummies_test = pd.get_dummies(test_df['Condition1'])

condition2_dummies_test = pd.get_dummies(test_df['Condition2'])

condition2_dummies_test['RRAn'] = 0

condition2_dummies_test['RRNn'] = 0

condition2_dummies_test['RRNe'] = 0

condition2_dummies_test['RRAe'] = 0

condition_dummies_test = condition1_dummies_test | condition2_dummies_test



street_dummies_test = pd.get_dummies(test_df['Street'])

lotConfig_dummies_test = pd.get_dummies(test_df['LotConfig'])

exterCond_dummies_test = pd.get_dummies(test_df['ExterCond'])

saleType_dummies_test = pd.get_dummies(test_df['SaleType'])

saleCond_dummies_test = pd.get_dummies(test_df['SaleCondition'])



bsmtCond_dummies_test = pd.get_dummies(test_df['BsmtCond'])

bsmtCond_dummies_test.columns = ['BC_Fa', 'BC_Gd', 'BC_Po', 'BC_TA']



heatingQC_dummies_test = pd.get_dummies(test_df['HeatingQC'])

heatingQC_dummies_test.columns = ['HQC_Ex', 'HQC_Fa', 'HQC_Gd', 'HQC_Po', 'HQC_TA']



kitchen_dummies_test = pd.get_dummies(test_df['KitchenQual'])

kitchen_dummies_test.columns = ['KQC_Ex', 'KQC_Fa', 'KQC_Gd', 'KQC_TA']



garageCond_dummies_test = pd.get_dummies(test_df['GarageCond'])

garageCond_dummies_test.columns = ['GC_Ex', 'GC_Fa', 'GC_Gd', 'GC_Po', 'GC_TA']





test_df = test_df.join(mszoning_dummies_test)

test_df.drop(['MSZoning'], axis=1, inplace=True)



test_df = test_df.join(bldgType_dummies_test)

test_df.drop(['BldgType'], axis=1, inplace=True)



test_df = test_df.join(houseStyle_dummies_test)

test_df.drop(['HouseStyle'], axis=1, inplace=True)



test_df = test_df.join(neigh_dummies_test)

test_df.drop(['Neighborhood'], axis=1, inplace=True)



test_df = test_df.join(condition_dummies_test)

test_df.drop(['Condition1'], axis=1, inplace=True)

test_df.drop(['Condition2'], axis=1, inplace=True)



test_df = test_df.join(street_dummies_test)

test_df.drop(['Street'], axis=1, inplace=True)



test_df = test_df.join(lotConfig_dummies_test)

test_df.drop(['LotConfig'], axis=1, inplace=True)



test_df = test_df.join(exterCond_dummies_test)

test_df.drop(['ExterCond'], axis=1, inplace=True)



test_df = test_df.join(saleType_dummies_test)

test_df.drop(['SaleType'], axis=1, inplace=True)



test_df = test_df.join(saleCond_dummies_test)

test_df.drop(['SaleCondition'], axis=1, inplace=True)



test_df = test_df.join(bsmtCond_dummies_test)

test_df.drop(['BsmtCond'], axis=1, inplace=True)



test_df = test_df.join(heatingQC_dummies_test)

test_df.drop(['HeatingQC'], axis=1, inplace=True)



test_df = test_df.join(kitchen_dummies_test)

test_df.drop(['KitchenQual'], axis=1, inplace=True)



test_df = test_df.join(garageCond_dummies_test)

test_df.drop(['GarageCond'], axis=1, inplace=True)

#test_df[['MSZoning', 'LotArea']].groupby(['MSZoning'],as_index=False).count()



#salePrice_df = train_df[['Id', 'SalePrice']].groupby('SalePrice').count()

#test_df = test_df.loc[test_df['LotArea']<20000]



#test_lotArea_median = test_df['LotArea'].median()

#test_df['LotArea'].loc[test_df['LotArea']>20000] = test_lotArea_median

#test_df['GrLivArea'].plot(kind='hist', figsize=(15,3),bins=100)
train_df.info()
test_df.info()
#train_df.count()

#train_df['LotArea'].loc[train_df['LotArea']>40000].count()



X_train = train_df.drop(['SalePrice'], axis=1)

Y_train = train_df['SalePrice']

X_test = test_df.copy()
#ridge_reg = linear_model.Ridge(alpha = 0.01)

#ridge_reg.fit(X_train, Y_train)

#Y_pred = ridge_reg.predict(X_test)

#ridge_reg.score(X_train, Y_train)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
#lasso_reg = linear_model.Lasso(alpha = 0.01)

#lasso_reg.fit(X_train, Y_train)

#Y_pred = lasso_reg.predict(X_test)

#lasso_reg.score(X_train, Y_train)
#elast_reg = linear_model.ElasticNet(alpha=0.01, l1_ratio=0.7)

#elast_reg.fit(X_train, Y_train)

#Y_pred = elast_reg.predict(X_test)

#elast_reg.score(X_train, Y_train)
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, Y_train)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)



print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")



imp_coef = pd.concat([coef.sort_values().head(20),

                     coef.sort_values().tail(20)])



import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label = Y_train)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":2, "eta":0.1}

xgb_model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, Y_train)
xgb_preds = np.expm1(model_xgb.predict(X_test))



lasso_preds = np.expm1(model_lasso.predict(X_test))
preds = 0.7*lasso_preds + 0.3*xgb_preds
scores = cross_validation.cross_val_score(lasso_reg, X_train, Y_train, cv=10)

scores
##random_forest = RandomForestClassifier(n_estimators=200)



#random_forest.fit(X_train, Y_train)



#Y_pred = random_forest.predict(X_test)



#random_forest.score(X_train, Y_train)
coeff_df = DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(lasso_reg.coef_)



# preview

coeff_df


#fig, (axis1, axis2) = plt.subplots(2,1,figsize=(20,8))



#train_df['newRooms'] = train_df['TotRmsAbvGrd']-train_df['TotalBathAbvGr']+train_df['TotalBathBsmt']-train_df['BedroomAbvGr']

#sns.pointplot(x='GrLivArea', y='SalePrice', data=train_df[['SalePrice', 'GrLivArea']].loc[train_df['GrLivArea'] <= 4000], ax=axis1)

#train_df['SalePrice'].plot(kind='line', ax=axis1)

#train_df['LotArea'].plot(kind='line', ax=axis1)

#train_df['GrLivArea'].plot(kind='line', ax=axis2)

#train_df['KitchenAbvGr'].plot(kind='line', ax=axis1, xlim=(0, 10))
submission = pd.DataFrame({

        "Id": test_df_Id,

        "SalePrice": preds

    })

submission.to_csv('HousePricePred_20170301_v2.csv', index=False)