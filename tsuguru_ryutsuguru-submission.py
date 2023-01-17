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

import matplotlib.pyplot as plt

import seaborn as sns





pd.set_option("display.max_columns", 50)
df = pd.read_csv("../input/exam-for-students20200527/train.csv")

df_submit = pd.read_csv("../input/exam-for-students20200527/sample_submission.csv")

df_city = pd.read_csv("../input/exam-for-students20200527/city_info.csv")

df_station = pd.read_csv("../input/exam-for-students20200527/station_info.csv")

df_test = pd.read_csv("../input/exam-for-students20200527/test.csv")
def add_distance_to_dfstation(df_staion):

    df_station.dropna(inplace=True)

    tokyo_station_Latitude = 35.681236

    tokyo_station_Longitude = 139.767125

    df_staion["distance"] = np.sqrt((df_station["Latitude"]-tokyo_station_Latitude)**2 +(df_station["Longitude"]-tokyo_station_Longitude)**2)

    

    return df_staion
def _change_type_numeric(df):

    df = df.replace({"Type":{"Residential Land(Land and Building)":0,

                                "Residential Land(Land Only)":1,

                                "Pre-owned Condominiums, etc.":2,

                                "Agricultural Land":3,

                                "Forest Land":4}})

    return df



def _add_chikunen(df):

    df["chikunen"] = df["Year"]-df["BuildingYear"]

    df = df.fillna({"chikunen":9999})

    return df



def _add_distance_to_df(df, df_staion):

    df = pd.merge(df,df_staion,left_on="NearestStation",right_on="Station",how="left")

    distance_mean = df["distance"].mean()

    df = df.fillna({"distance":distance_mean})

    return df



def _purpose_numeric(df):

    df = df.replace({"Purpose":{"House":74323,"Other":8325,"Office":1443,"Shop":1237,

                               "Warehouse":559,"Factory":367}})

    df = df.fillna({"Purpose":0})

    return df



def _region_numeric(df):

    df = df.replace({"Region":{"Residential Area":151204,"Commercial Area":11070,

                               "Potential Residential Area":5983,"Industrial Area":1787}})

    df = df.fillna({"Region":0})

    df["Region"]=df["Region"].astype("int")

    return df



def _renovation_numeric(df):

    df = df.replace({"Renovation":{"Not yet":47452,"Done":18505}})

    df = df.fillna({"Renovation":0})

    return df





def preprocess(df,df_station_):

    df = _change_type_numeric(df)

    df = _add_chikunen(df)

    df = _add_distance_to_df(df, df_station_)

    df = _purpose_numeric(df)

    df = _region_numeric(df)

    df = _renovation_numeric(df)

    

    return df
#stationデータ前処理

df_station_ = add_distance_to_dfstation(df_station)
df = preprocess(df,df_station_)
df.head()
df.columns
use_columns = ["Type", "Region","MinTimeToNearestStation","Area","AreaIsGreaterFlag",

              "PrewarBuilding","Purpose","chikunen","distance"]
import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=123)
df_X = df[use_columns]

df_y = df["TradePrice"]
train_X, test_X, train_y, test_y = train_test_split(df_X, df_y, test_size=0.2,random_state=123)



train_y = np.log1p(train_y)

test_y = np.log1p(test_y)
lgb_train = lgb.Dataset(train_X, train_y)

lgb_test = lgb.Dataset(test_X, test_y, reference=lgb_train)
param_grid = {"max_depth":[5,10,25,50,75,100],

             "learning_rate":[0.001,0.01,0.05,0.1],

             "num_leaves":[100,300,500,1000],

             }
params = {                                                                                                                                                                         

    'objective': 'regression',                                                                           

    'metric': 'l2',                                                                             

#    'num_leaves': 40,                                                                                    

    'learning_rate': 0.1,                                                                               

    'feature_fraction': 0.9,                                                                             

    'bagging_fraction': 0.8,                                                                             

    'bagging_freq': 5, 

}                                                                                                        

                                                                                                         

model = lgb.train(params,                                                                                  

                lgb_train,                                                                               

                num_boost_round=200,                                                                      

                valid_sets=lgb_test,                                                                     

                early_stopping_rounds=10

                ) 

#clf = lgb.LGBMRegressor()

#grid_search = GridSearchCV(clf,param_grid=param_grid,cv=5)



#grid_search.fit(train_X,train_y)
y_pred = model.predict(test_X, num_iteration=model.best_iteration)

#y_pred = np.exp(y_pred)-1
plt.plot((test_y-y_pred), "o",ms=1)
np.sum(np.power((y_pred-test_y),2))/len(test_y)
df_test = preprocess(df_test, df_station_)
df_test_X = df_test[use_columns]
y_test_pred = model.predict(df_test_X, num_iteration=model.best_iteration)

y_test_pred = np.exp(y_test_pred)-1
y_test_pred
df_pred = pd.DataFrame(y_test_pred,columns=["TradePrice"])

df_submit_ = pd.concat([df_test,df_pred],axis=1)

df_submit_ = df_submit_[["id","TradePrice"]]
df_submit_.head()
df_submit_
df_submit_.to_csv("submission.csv",index=False)
df_submit_.shape,df_submit.shape