#Essentials

import numpy as np

import pandas as pd



#Plotting

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')





#Models

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.model_selection import KFold, cross_val_score

from mlxtend.regressor import StackingRegressor





#Other

from scipy.stats import norm

from sklearn.model_selection import GridSearchCV
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_train.shape, df_test.shape
train_ids = df_train['Id']

test_ids = df_test['Id']

df_train.drop('Id', axis=1, inplace=True)

df_test.drop('Id', axis=1, inplace=True)

df_train.shape, df_test.shape
f, ax = plt.subplots(figsize=(8, 7))

sns.set_style("white")

sns.distplot(df_train.SalePrice)
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
f, ax = plt.subplots(figsize=(10, 7))

sns.set_style("white")

sns.set_color_codes(palette='deep')

sns.distplot(df_train.SalePrice, fit=norm, color='b')

mu, sigma = norm.fit(df_train.SalePrice)

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
df_train_labels = df_train['SalePrice'].reset_index(drop=True)

df_train_features = df_train.drop(['SalePrice'], axis=1)

df_test_features = df_test



df_all_features = pd.concat([df_train_features, df_test_features]).reset_index(drop=True)

df_all_features.shape
def handle_missing(features):

    

    features['Functional'] = features['Functional'].fillna('Typ')

    features['Electrical'] = features['Electrical'].fillna("SBrkr")

    features['KitchenQual'] = features['KitchenQual'].fillna("TA")

    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    

    features["PoolQC"] = features["PoolQC"].fillna("None")

    

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

        features[col] = features[col].fillna(0)

    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

        features[col] = features[col].fillna('None')

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

        features[col] = features[col].fillna('None')

        

    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



    objects = []

    for i in features.columns:

        if features[i].dtype == object:

            objects.append(i)

    features.update(features[objects].fillna('None'))

        

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    numeric = []

    for i in features.columns:

        if features[i].dtype in numeric_dtypes:

            numeric.append(i)

    features.update(features[numeric].fillna(0))    

    return features



df_all_features = handle_missing(df_all_features)
df_all_features = pd.get_dummies(df_all_features)
X = df_all_features.iloc[:len(df_train_labels), :]

X_test = df_all_features.iloc[len(df_train_labels):, :]

X.shape, df_train_labels.shape, X_test.shape
kf = KFold(n_splits=10, random_state=42, shuffle=True)
# Инициируем модель RandomForest

rf = RandomForestRegressor(n_estimators=1200,

                          max_depth=15,

                          min_samples_split=5,

                          min_samples_leaf=5,

                          max_features=None,

                          oob_score=True,

                          random_state=42)
rf_model = rf.fit(X, df_train_labels)
rf_feature_importance = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['feature_importance']).sort_values('feature_importance', ascending=False)

rf_feature_importance[rf_feature_importance.feature_importance > 0.01].plot(kind="bar")
# инициируем еще несколько моделей для последующего стэкинга



lr = LinearRegression()



rd = Ridge()



gb = GradientBoostingRegressor(n_estimators = 40, max_depth = 2)
model = StackingRegressor(

    regressors=[rf, gb, rd],

    meta_regressor=lr

)



# Обучаем модель

model.fit(X, df_train_labels)
# Напишем формулу для оценки точности модели



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, df_train_labels, scoring="neg_mean_squared_error", cv=kf))

    return (np.mean(rmse))
cv_rmse(model)
cv_rmse(rf_model)
rd_model = rd.fit(X, df_train_labels)

cv_rmse(rd_model)
lr_model = lr.fit(X, df_train_labels)

cv_rmse(lr_model)