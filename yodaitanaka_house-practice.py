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
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train
test_ID = test['Id']
train = train.drop("Id",axis=1)
X_train=train.drop("SalePrice",axis=1)

y_train=train["SalePrice"]
import seaborn as sns

import matplotlib.pyplot as plt

sns.distplot(y_train)

plt.show()
import seaborn as sns

import matplotlib.pyplot as plt

y_train = np.log(y_train)



sns.distplot(y_train)

plt.show()
all_data=pd.concat([X_train,test],axis=0,sort=True)
na_col_list=all_data.isnull().sum()[all_data.isnull().sum()>0].index.tolist()
all_data[na_col_list].dtypes.sort_values()
#隣接した道路の長さ（LotFrontage）の欠損値の補完

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



#欠損値が存在するかつfloat型のリストを作成

float_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "float64"].index.tolist()



#欠損値が存在するかつobject型のリストを作成

obj_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "object"].index.tolist()



#float型の場合は欠損値を0で置換

all_data[float_list] = all_data[float_list].fillna(0)



#object型の場合は欠損値を"None"で置換

all_data[obj_list] = all_data[obj_list].fillna("None")



#欠損値が全て置換できているか確認

all_data.isnull().sum()[all_data.isnull().sum() > 0]
# カテゴリ変数に変換する

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
#家全体の面積

all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]



#特徴量に1部屋あたりの面積を追加

all_data["FeetPerRoom"] = all_data["TotalSF"]/all_data["TotRmsAbvGrd"]



#建築した年とリフォームした年の合計

all_data['YearBuiltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']



#バスルームの合計面積

all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +

                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))



#縁側の合計面積

all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] +

                              all_data['EnclosedPorch'] + all_data['ScreenPorch'] +

                              all_data['WoodDeckSF'])

#プールの有無

all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)



#2階の有無

all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)



#ガレージの有無

all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)



#地下室の有無

all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)



#暖炉の有無

all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
#カテゴリ変数となっているカラムを取り出す

cal_list = all_data.dtypes[all_data.dtypes=="object"].index.tolist()



#学習データにおけるカテゴリ変数のデータ数を確認

X_train[cal_list].info()
#カテゴリ変数をget_dummiesによるone-hot-encodingを行う

all_data = pd.get_dummies(all_data,columns=cal_list)



#サイズを確認

all_data.shape
# trainデータとtestデータを含んでいるXmatを、再度trainデータとtestデータに分割

X_train = all_data.iloc[:train.shape[0],:]

X_test = all_data.iloc[train.shape[0]:,:]



# ランダムフォレストをインポート

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, max_features='auto')

rf.fit(X_train, y_train)



# np.argsort()はソート結果の配列のインデックスを返す。引数の頭に"-"をつけると降順。

# つまり"-rf.feature_importances_"を引数にする事で重要度の高い順にソートした配列のインデックスを返す。

ranking = np.argsort(-rf.feature_importances_)



#重要度の降順で特徴量の名称、重要度を表示

for f in range(50):

    print("%2d) %-*s %f" % (f + 1, 50, 

                            X_train.columns.values[ranking[f]], 

                            rf.feature_importances_[ranking[f]]))
#上位30個を特徴量として扱う

X_train = X_train.iloc[:,ranking[:30]]

X_test = X_test.iloc[:,ranking[:30]]
X_train = (X_train - X_train.mean()) / X_train.std()

X_test = (X_test - X_test.mean()) / X_test.std()
import matplotlib.pyplot as plt

import seaborn as sns

fig = plt.figure(figsize=(12,7))

for i in np.arange(30):

    ax = fig.add_subplot(5,6,i+1)

    sns.regplot(x=X_train.iloc[:,i], y=y_train)



plt.tight_layout()

plt.show()
all_data = X_train

all_data['SalePrice'] = y_train

all_data = all_data.drop(index = all_data[(all_data['TotalSF'] > 5) & (all_data['SalePrice'] < 12.5)].index)

all_data = all_data.drop(index = all_data[(all_data['BsmtFinSF1'] > 10) & (all_data['SalePrice'] < 13)].index)

all_data = all_data.drop(index = all_data[(all_data['GrLivArea'] > 5) & (all_data['SalePrice'] < 13)].index)

all_data = all_data.drop(index = all_data[(all_data['FeetPerRoom'] > 5) & (all_data['SalePrice'] < 13)].index)

all_data = all_data.drop(index = all_data[(all_data['1stFlrSF'] > 5) & (all_data['SalePrice'] < 13)].index)

all_data = all_data.drop(index = all_data[(all_data['TotalBsmtSF'] > 10) & (all_data['SalePrice'] < 13)].index)

# recover

y_train = all_data['SalePrice']

X_train = all_data.drop(['SalePrice'], axis=1)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=12, random_state=42, shuffle=True)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
# Define error metrics

from sklearn.metrics import mean_squared_error

def rmsle(y_valid, y_pred):

    return np.sqrt(mean_squared_error(y_valid, y_pred))



def cv_rmse(model):

    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf))

    return (rmse)
rf = RandomForestRegressor(n_estimators=1200,

                          max_depth=15,

                          min_samples_split=5,

                          min_samples_leaf=5,

                          max_features=None,

                          oob_score=True)
rf_li = []

for i in range(10):

    score = cv_rmse(rf)

    rf_li.append(score)

print("rf: {:.4f} ({:.4f})".format(np.mean(rf_li), np.std(rf_li)))
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVR

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))
svr_li = []

for i in range(10):

    score = cv_rmse(svr)

    svr_li.append(score)

print("rf: {:.4f} ({:.4f})".format(np.mean(svr_li), np.std(svr_li)))
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=6000,

                                learning_rate=0.01,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber')
gbr_li = []

for i in range(10):

    score = cv_rmse(gbr)

    gbr_li.append(score)

print("rf: {:.4f} ({:.4f})".format(np.mean(gbr_li), np.std(gbr_li)))
import optuna

from sklearn.metrics import mean_squared_error

import lightgbm as lgb



def objective(trial):

    params = {'objective': 'regression',

              'metric': {'rmse'},

              'max_depth' : trial.suggest_int('max_depth', 1, 10),

              'subsumple' : trial.suggest_uniform('subsumple', 0.0, 1.0),

              'subsample_freq' : trial.suggest_int('subsample_freq', 0, 1),

              'leaning_rate' : trial.suggest_loguniform('leaning_rate', 1e-5, 1),

              'feature_fraction' : trial.suggest_uniform('feature_fraction', 0.0, 1.0),

              'lambda_l1' : trial.suggest_uniform('lambda_l1' , 0.0, 1.0),

              'lambda_l2' : trial.suggest_uniform('lambda_l2' , 0.0, 1.0)}

    

    lgb_train = lgb.Dataset(X_train, y_train)

    lgb_eval = lgb.Dataset(X_valid, y_valid)



    model = lgb.train(

        params, lgb_train,

        valid_sets=[lgb_train, lgb_eval],

        verbose_eval=100,

        num_boost_round=20000,

        early_stopping_rounds=100

    )



    predicted = model.predict(X_valid)

    RMSE = np.sqrt(mean_squared_error(y_valid, predicted))

    return RMSE
study = optuna.create_study()

optuna.logging.disable_default_handler()

study.optimize(objective, n_trials=1000)
params = {'objective': 'regression',

          'metric': {'rmse'},

          'max_depth' : study.best_params['max_depth'],

          'subsumple' : study.best_params['subsumple'],

          'subsample_freq' : study.best_params['subsample_freq'],

          'leaning_rate' : study.best_params['leaning_rate'],

          'feature_fraction' : study.best_params['feature_fraction'],

          'lambda_l1' : study.best_params['lambda_l1'],

          'lambda_l2' : study.best_params['lambda_l2']}

lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_valid, y_valid)



model = lgb.train(

    params, lgb_train,

    valid_sets=[lgb_train, lgb_eval],

    verbose_eval=100,

    num_boost_round=20000,

    early_stopping_rounds=100

)



y_pred_lgbm = model.predict(X_valid, num_iteration=model.best_iteration)
y3= 100

best_param={}

for x1 in np.arange(0,1.0,0.1):

    for x2 in np.arange(0,1-x1,0.1):

        for x3 in np.arange(0,1-x1-x2,0.1):

            for x4 in np.arange(0,1-x1-x2-x3,0.1):

                y_pred_all = (y_pred_rf*x1+ y_pred_svr * x2+ y_pred_gbr * x3+y_pred_lgbm * x4)/4

                y1 = (np.log(mean_squared_error(y_valid, y_pred_all)))

                best_param[y1] = [x1,x2,x3,x4]

                y3 = min(y3,y1)

print(y3)

print(best_param[y3])
from lightgbm import LGBMRegressor

lightgbm = LGBMRegressor(objective='regression',

                         max_depth = study.best_params['max_depth'],

                         subsumple = study.best_params['subsumple'],

                         subsample_freq = study.best_params['subsample_freq'],

                         leaning_rate = study.best_params['leaning_rate'],

                         feature_fraction = study.best_params['feature_fraction'],

                         lambda_l1 = study.best_params['lambda_l1'],

                         lambda_l2 = study.best_params['lambda_l2'],

                         verbose=-1)
params = {'objective': 'regression',

          'metric': {'rmse'},

          'max_depth' : study.best_params['max_depth'],

          'subsumple' : study.best_params['subsumple'],

          'subsample_freq' : study.best_params['subsample_freq'],

          'leaning_rate' : study.best_params['leaning_rate'],

          'feature_fraction' : study.best_params['feature_fraction'],

          'lambda_l1' : study.best_params['lambda_l1'],

          'lambda_l2' : study.best_params['lambda_l2']}

lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_valid, y_valid)



model = lgb.train(

    params, lgb_train,

    valid_sets=[lgb_train, lgb_eval],

    verbose_eval=100,

    num_boost_round=20000,

    early_stopping_rounds=100

)



y_pred_lgbm = model.predict(X_test, num_iteration=model.best_iteration)
lgb_li = []

for i in range(10):

    score = cv_rmse(lightgbm)

    lgb_li.append(score)

print("lgb: {:.4f} ({:.4f})".format(np.mean(lgb_li), np.std(lgb_li)))
X_train=pd.concat([X_train,X_valid],axis=0,sort=True)
y_train=pd.concat([y_train,y_valid],axis=0,sort=True)
#学習の実行

rf.fit(X_train, y_train)

#テストデータで予測実行

y_pred_rf = rf.predict(X_test)
#学習の実行

svr.fit(X_train, y_train)

#テストデータで予測実行

y_pred_svr = svr.predict(X_test)
#学習の実行

gbr.fit(X_train, y_train)

#テストデータで予測実行

y_pred_gbr = gbr.predict(X_test)
y_pred = y_pred_rf*best_param[y3][0] + y_pred_svr*best_param[y3][1] + y_pred_gbr * best_param[y3][2] + y_pred_lgbm * best_param[y3][3]
y_pred = np.floor(np.expm1(y_pred_lgbm))
submission = pd.DataFrame({

    "Id": test_ID,

    "SalePrice": y_pred

})

submission.to_csv('submission.csv', index=False)