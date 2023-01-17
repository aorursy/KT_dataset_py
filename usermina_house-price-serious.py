import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV





import xgboost as xgb

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

import data_analyze as DataAnalyze

# My script

if _dh == ['/kaggle/working']:

    test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

    train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

else:

    train = pd.read_csv('./data/train.csv')

    test = pd.read_csv('./data/test.csv')
train
x_train=train.drop("SalePrice",axis = 1)

y_train = train.SalePrice

all_data=pd.concat([x_train, test])

all_data
train_Id = train['Id']

test_Id= test['Id']
# Id drop

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
train
DataAnalyze.make_missing_table(all_data)
DataAnalyze.make_missing_table(train)
DataAnalyze.make_missing_table(test)
# SalePrice

sns.distplot(y_train)
# y_train log

y_train=np.log(y_train)

sns.distplot(y_train)
# Drop



# NoSeWa train only

all_data = all_data.drop(['Utilities'], axis=1)
# Completion miss



# LotFrontage median of Neighborhood group

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



# mode

all_data = DataAnalyze.missing_mode_completion(all_data,["MSZoning","Electrical","KitchenQual","Exterior1st","Exterior2nd","SaleType"])



# Fill None 

none_list=["PoolQC","MiscFeature","Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond","Functional","MSSubClass"]

all_data=DataAnalyze.missing_None_completion(all_data, none_list)



# int float zero completion

all_data=DataAnalyze.missing_zero_completion(all_data)

# all_data = all_data.fillna(all_data.median())

all_data
# covert int to category　



# MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

# Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

# Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
# LabelEncode

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))

all_data# dummy

all_data=pd.get_dummies(all_data, drop_first=True, dummy_na=True)

all_data
# dummy

all_data=pd.get_dummies(all_data, drop_first=True, dummy_na=True)

all_data
# new feature

all_data["TotalSF"]=all_data["TotalBsmtSF"]+all_data["1stFlrSF"]+all_data["2ndFlrSF"]

# print(all_data.columns)
# Id<1460 is train

x_train=all_data[all_data["Id"] <= 1460]

x_test=all_data[all_data["Id"] >1460]

x_train=x_train.drop("Id",axis = 1)

x_test=x_test.drop("Id",axis = 1)
# zscore norm.

x_train=DataAnalyze.zscore_normalization_describe(x_train,all_data.describe())

x_test=DataAnalyze.zscore_normalization_describe(x_test,all_data.describe())
# feature importance using random forest

DataAnalyze.plot_feature_importances(x_train, y_train, num=30, n_estimators=80)
DataAnalyze.make_feature_importances_table(x_train, y_train, num=20, n_estimators=80)
# Top 30

feature_importances_table=DataAnalyze.make_feature_importances_table(x_train, y_train, num=30, n_estimators=80)

x_train=x_train.loc[:,feature_importances_table["collumn"].tolist()]

x_train
# test

x_test=x_test.loc[:,feature_importances_table["collumn"].tolist()]

x_test
# relation to the target

DataAnalyze.prot_relation_target(x_train, y_train)
# outlier data deletion



Xmat = x_train

Xmat['SalePrice'] = y_train

Xmat = Xmat.drop(Xmat[(Xmat['TotalSF']>5) & (Xmat['SalePrice']<13)].index)

Xmat = Xmat.drop(Xmat[(Xmat['GrLivArea']>5) & (Xmat['SalePrice']<13)].index)

Xmat = Xmat.drop(Xmat[(Xmat['OpenPorchSF']>6) & (Xmat['SalePrice']<11)].index)



y_train = Xmat['SalePrice']

x_train = Xmat.drop(['SalePrice'], axis=1)

x_train
DataAnalyze.prot_relation_target(x_train, y_train)
# xgb

# params={'max_depth': [2,4,6],'n_estimators': [100,200,300]}

params={'max_depth': [2],'n_estimators': [200]}

xgb_model = xgb.XGBRegressor()

reg_xgb = GridSearchCV(xgb_model,params,return_train_score=False,cv=2,verbose=1)

reg_xgb.fit(x_train, y_train)

pd.DataFrame.from_dict(reg_xgb.cv_results_)
# NN

def create_model(optimizer='adam',mid=16):

    model = Sequential()

    model.add(Dense(x_train.shape[1], input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))

    model.add(Dense(16, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))



    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model



model = KerasRegressor(build_fn=create_model, verbose=0)

# define the grid search parameters

# params={'optimizer': ["adam"],'batch_size': [4,8,16],"epochs":[100,200,300]}

params={'optimizer': ["adam"],'batch_size': [4],"epochs":[200]}

reg_dl = GridSearchCV(estimator=model, param_grid=params, cv=2,n_jobs=-1)

reg_dl.fit(x_train, y_train)

pd.DataFrame.from_dict(reg_dl.cv_results_)
test_pred_df = pd.DataFrame( {'XGB': reg_xgb.predict(x_test),

                              'DL': reg_dl.predict(x_test).ravel()});

test_pred_df
train_pred_df=pd.DataFrame( {'XGB': reg_xgb.predict(x_train),

                             'DL': reg_dl.predict(x_train).ravel()});

train_pred_df
# ansanble

from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit(train_pred_df, y_train)



ansan_pred = reg.predict(test_pred_df)

ansan_pred
# y_pred = np.exp(test_pred_df["XGB"])

# y_pred = np.exp(test_pred_df["DL"])

y_pred = np.exp(ansan_pred)

submission = pd.DataFrame({

    "Id": test_Id,

    "SalePrice": y_pred

})

# submission.to_csv("output_xgb.csv",index=False)

# submission.to_csv("output_dl.csv",index=False)

submission.to_csv("output_ansan.csv",index=False)

submission