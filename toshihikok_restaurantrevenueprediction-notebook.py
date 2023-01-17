#%%

import pandas as pd

import numpy as np

import datetime

from sklearn.preprocessing import LabelEncoder



# from sklearn.svm import SVC

from sklearn.metrics import classification_report

from sklearn import metrics

# from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

import seaborn as sb





#%%

#データを読み込んでマージする

train = pd.read_csv("../input/restaurant-revenue-prediction/train.csv.zip")

test = pd.read_csv("../input/restaurant-revenue-prediction/test.csv.zip")





#%%

#City_data = pd.read_csv("CityList_input.csv")

#City_data = City_data.drop('ID', axis=1)

#CityData = City_data.rename(columns = {'City_name':'City'})

#CityData





#%%



acc_dic = {}



train['WhatIsData'] = 'Train'

test['WhatIsData'] = 'Test'

test['revenue'] = 9999999999

alldata = pd.concat([train,test],axis=0).reset_index(drop=True)





#Cityを数値に変換

le = LabelEncoder()

alldata["City"] = le.fit_transform(alldata["City"])



# City Groupを数値に変換 Other -> 0, Big Cities -> 1

alldata["City Group"] = alldata["City Group"].map({"Other":0, "Big Cities":1})





alldata["Open Date"] = pd.to_datetime(alldata["Open Date"])

alldata["Year"] = alldata["Open Date"].apply(lambda x:x.year)

alldata["Month"] = alldata["Open Date"].apply(lambda x:x.month)

alldata["Day"] = alldata["Open Date"].apply(lambda x:x.day)

alldata["kijun"] = "2015-04-27"

alldata["kijun"] = pd.to_datetime(alldata["kijun"])

alldata["BusinessPeriod"] = (alldata["kijun"] - alldata["Open Date"]).apply(lambda x: x.days)

#alldata = pd.merge(alldata, CityData, how="left", on="City")



alldata = alldata.drop('Open Date', axis=1)

alldata = alldata.drop('kijun', axis=1)

#alldata = alldata.drop('City', axis=1)

#alldata = alldata.drop('Area', axis=1)

#alldata = alldata.drop('Density', axis=1)



#%%

#sb.relplot(x="Open Date", y="revenue", col="City Group", data=train)



#%%



# 訓練データ特徴量をリスト化

cat_cols = alldata.dtypes[alldata.dtypes=='object'].index.tolist()

num_cols = alldata.dtypes[alldata.dtypes!='object'].index.tolist()



other_cols = ['Id','WhatIsData']

# 余計な要素をリストから削除

cat_cols.remove('WhatIsData') #学習データ・テストデータ区別フラグ除去

num_cols.remove('Id') #Id削除



# カテゴリカル変数をダミー化

cat = pd.get_dummies(alldata[cat_cols])



# データ統合

all_data = pd.concat([alldata[other_cols],alldata[num_cols].fillna(0),cat],axis=1)



# plt.hist(np.log(train['revenue']), bins=50)

# plt.hist(train['revenue'], bins=50)



train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)

test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','revenue'], axis=1).reset_index(drop=True)



x_ = train_.drop('revenue',axis=1)

y_ = train_.loc[:, ['revenue']]

y_ = np.log(y_)

test_feature = test_.drop('Id',axis=1)



X_train, X_test, y_train, y_test = train_test_split(

    x_, y_, random_state=0, train_size=0.9,shuffle=True)



#%%

# サンプルから欠損値と割合、データ型を調べる関数

def Missing_table(df):

    null_val = df.isnull().sum()

    # null_val = df.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)

    percent = 100 * null_val/len(df)

    # list_type = df.isnull().sum().dtypes #データ型

    Missing_table = pd.concat([null_val, percent], axis = 1)

    missing_table_len = Missing_table.rename(

    columns = {0:'欠損値', 1:'%', 2:'type'})

    return missing_table_len.sort_values(by=['欠損値'], ascending=False)



Missing_table(train)



#%%

# サンプルからデータ型を調べる関数

#def Datatype_table(df):

#        list_type = df.dtypes #データ型

#        Datatype_table = pd.concat([list_type], axis = 1)

#        Datatype_table_len = Datatype_table.rename(columns = {0:'データ型'})

#        return Datatype_table_len

#    

#Datatype_table(City_data)



#%%

test.describe()



#%%

# lightGBMによる予測

lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)



# LightGBM parameters

params = {

        'task' : 'train',

        'boosting_type' : 'gbdt',

        'objective' : 'regression',

        'metric' : {'l2'},

        'num_leaves' : 31,

        'learning_rate' : 0.1,

        'feature_fraction' : 0.9,

        'bagging_fraction' : 0.8,

        'bagging_freq': 5,

        'verbose' : 0,

        'n_jobs': 2

}



gbm = lgb.train(params,

            lgb_train,

            num_boost_round=100,

            valid_sets=lgb_eval,

            early_stopping_rounds=10)



# cv_results = lgb.cv(params,

#                 lgb_train,

#                 num_boost_round=100,

#                 early_stopping_rounds=10,

#                 nfold=5,

#                 shuffle=True,

#                 stratified=False,

#                 seed=42)



# np.mean(cv_results["l2-mean"])



# prediction_train = gbm.predict(X_train)



# y_pred = []

# for x in prediction_train:

#         y_pred.append(x)



# y_true = y_train['revenue'].tolist()



# acc_lightGBM =  mean_squared_error(y_true, np.exp(y_pred))

# acc_dic.update(model_lightGBM = round(acc_lightGBM,3))



prediction_lgb = np.exp(gbm.predict(test_feature))





#%%

# RandomForestRegressorによる予測

forest = RandomForestRegressor().fit(X_train, y_train)

prediction_rf = np.exp(forest.predict(test_feature))



acc_forest = forest.score(X_train, y_train)

acc_dic.update(model_forest = round(acc_forest,3))

print(f"training dataに対しての精度: {forest.score(X_train, y_train):.2}")



#%%

# lasso回帰による予測

lasso = Lasso().fit(X_train, y_train)

prediction_lasso = np.exp(lasso.predict(test_feature))



acc_lasso = lasso.score(X_train, y_train)

acc_dic.update(model_lasso = round(acc_lasso,3))

print(f"training dataに対しての精度: {lasso.score(X_train, y_train):.2}")



#%%

# ElasticNetによる予測

En = ElasticNet().fit(X_train, y_train)

prediction_en = np.exp(En.predict(test_feature))

print(f"training dataに対しての精度: {En.score(X_train, y_train):.2}")



acc_ElasticNet = En.score(X_train, y_train)

acc_dic.update(model_ElasticNet = round(acc_ElasticNet,3))



#%%

# ElasticNetによるパラメータチューニング

parameters = {

        'alpha'      : [0.001, 0.01, 0.1, 1, 10, 100],

        'l1_ratio'   : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],

}



En2 = GridSearchCV(ElasticNet(), parameters)

En2.fit(X_train, y_train)

prediction_en2 = np.exp(En.predict(test_feature))



acc_ElasticNet_Gs = En2.score(X_train, y_train)

acc_dic.update(model_ElasticNet_Gs = round(acc_ElasticNet_Gs,3))

print(f"training dataに対しての精度: {En.score(X_train, y_train):.2}")





#%%

# 各モデルの訓練データに対する精度をDataFrame化

Acc = pd.DataFrame([], columns=acc_dic.keys())

dict_array = []

for i in acc_dic.items():

        dict_array.append(acc_dic)

Acc = pd.concat([Acc, pd.DataFrame.from_dict(dict_array)]).T

Acc[0]



#%%

# Idを取得

Id = np.array(test["Id"]).astype(int)

# 予測データとIdをデータフレームへ落とし込む

result = pd.DataFrame(prediction_lgb, Id, columns = ["Prediction"])

# csvとして書き出し

result.to_csv("/kaggle/working/submission.csv", index_label = ["Id"])