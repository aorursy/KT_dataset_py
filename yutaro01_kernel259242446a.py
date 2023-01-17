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



df_train=pd.read_csv('../input/exam-for-students20200527/train.csv')

df_test=pd.read_csv('../input/exam-for-students20200527/test.csv')

df_train['is_train']=1

df_test['is_train']=0



city=pd.read_csv('../input/exam-for-students20200527/city_info.csv')

station=pd.read_csv('../input/exam-for-students20200527/station_info.csv')



target='TradePrice'



df=pd.concat([df_train,df_test])
#駅、住所の緯度経度を結合

df=pd.merge(df,station,left_on='NearestStation',right_on='Station',how='left')

df=pd.merge(df,city,on=['Prefecture','Municipality'],how='left')

df.columns=['id', 'Type', 'Region', 'Prefecture', 'Municipality', 'DistrictName',

       'NearestStation', 'TimeToNearestStation', 'MinTimeToNearestStation',

       'MaxTimeToNearestStation', 'FloorPlan', 'Area', 'AreaIsGreaterFlag',

       'LandShape', 'Frontage', 'FrontageIsGreaterFlag', 'TotalFloorArea',

       'TotalFloorAreaIsGreaterFlag', 'BuildingYear', 'PrewarBuilding',

       'Structure', 'Use', 'Purpose', 'Direction', 'Classification', 'Breadth',

       'CityPlanning', 'CoverageRatio', 'FloorAreaRatio', 'Year', 'Quarter',

       'Renovation', 'Remarks', 'TradePrice', 'is_train', 'Station',

       'Latitude_st', 'Longitude_st', 'Latitude_ct', 'Longitude_ct']



atodekesu=['Prefecture', 'Municipality','Station','TimeToNearestStation','id',

           'PrewarBuilding','FrontageIsGreaterFlag','TotalFloorAreaIsGreaterFlag']
# df.CityPlanning.value_counts()

df.Remarks.value_counts()
tokai=['Tkyo','Ikebukuro','Omiya','Shinjuku','Shinagawa','Kawaguchi','Urawa','Kawagoe','Kumagaya','Tokorozawa','Shiki','Soka','Kasukabe','Saitamashintoshin']

# station.query('Station=="Otemachi"')

station
df.query('Prefecture=="Tokyo"')['NearestStation'].drop_duplicates()
def kyori(x,ido,kei):

    return np.sqrt((x.Latitude_st-ido)**2+(x.Longitude_st-kei)**2)



tokai=['Ikebukuro','Tokyo','Shinagawa','Shinjuku','Shibuya','Akihabara','Iidabashi','Kitasenju','Osaki','Hachioji','Shinbashi']

for i in tokai:

    ido=station.query('Station==@i')['Latitude'].values[0]

    kei=station.query('Station==@i')['Longitude'].values[0]

    df['from_'+i]=df[['Latitude_st','Longitude_st']].apply(lambda x:kyori(x,ido,kei),axis=1)

    print(i)

df.head()
pd.set_option('display.max_columns', 1000)

pd.set_option('display.max_rows', 700)
def nummcount(x):

    leng=len(x.split(','))

    if x=='No_use':

        leng=0

    return leng

#ターゲット変数のlog化

df[target]=df[target].apply(np.log1p)

df['MaxTimeToNearestStation']=df['MaxTimeToNearestStation'].map(lambda x: 9999 if x!=x else x)

df['FrontageIsGreaterFlag']=df['FrontageIsGreaterFlag'].astype(str)

df.Use=df.Use.fillna('No_use')

df['use_num']=df.Use.map(lambda x: nummcount(x))

df['spent_bldyear']=df.BuildingYear.map(lambda x: 2020 - x)
df.query('TotalFloorAreaIsGreaterFlag==1')[['TotalFloorArea','TotalFloorAreaIsGreaterFlag','AreaIsGreaterFlag']].head(200)

df['DistrictName'].value_counts().head(100)
#余計なデータ消去

df=df.drop(atodekesu,axis=1)



#欠損処理

df=df.fillna(df.median())



#データ分割

X_train = df.query('is_train==1').drop(['is_train',target],axis=1)

y_train = df.query('is_train==1')[target]

X_test = df.query('is_train==0').drop(['is_train',target],axis=1)



cats=X_train.select_dtypes(include=[object]).columns



#OrdinalEncoderを使う場合カテゴリデータのエンコーディング

from category_encoders import OrdinalEncoder, OneHotEncoder



oe = OrdinalEncoder(cols=cats, return_df=False)

X_train[cats] = oe.fit_transform(X_train[cats])

X_test[cats] = oe.transform(X_test[cats])
X_train.head()
import lightgbm as lgb

from lightgbm import LGBMRegressor

from sklearn.linear_model import Ridge



lgbm = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1, learning_rate=0.1,

                     n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None,

                     min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0,

                     subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,

                     random_state=None, n_jobs=- 1, silent=True, importance_type='split')
from sklearn.model_selection import KFold



# yの値がlog変換されていること前提

def rmsle(y,y_pred):

    assert y.shape == y_pred.shape

    return np.sqrt(np.mean(np.square(y_pred - y)))



ridge_socores=[]

lgbm_scores=[]

y_pred_lgbm=pd.DataFrame()

y_pred_ridge=pd.DataFrame()



k=5

skf = KFold(n_splits=k, random_state=71, shuffle=True)

for i, (train_ix, test_ix) in enumerate(skf.split(X_train, y_train)):

    print('fold = ',i)

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix] 



    #学習 LightGBM

    lgbm.fit(X_train_, y_train_,early_stopping_rounds=20,eval_metric='RMSE',eval_set=[(X_val,y_val)])

    lgbm_scores.append(rmsle(y_val,lgbm.predict(X_val))) #(正解データ, 予測値)

    y_pred_lgbm=pd.concat([y_pred_lgbm,pd.DataFrame(lgbm.predict(X_test,num_iteration =lgbm.best_iteration_))],axis=1)



#     #学習 Ridge

#     ridge.fit(X_train_, y_train_)

#     ridge_scores.append(rmsle(y_val,ridge.predict(X_val))) #(正解データ, 予測値)

#     y_pred_ridge=pd.concat([y_pred_ridge,pd.DataFrame(ridge.predict(X_test))],axis=1)



print('LightGBM RMSLE:',lgbm_scores, 'Avg:',sum(lgbm_scores)/k)

# print('Ridge    RMSLE:',ridge_scores, 'Avg:',sum(ridge_scores)/k)
print('LightGBM RMSLE:',lgbm_scores, 'Avg:',sum(lgbm_scores)/k)
lgbm.feature_importances_

pd.DataFrame(lgbm.feature_importances_, index=X_train.columns, columns=['importance']).sort_values(by='importance')
#各モデルの結果平均を算出し、足し合わせる

y_pred_lgbm['avg']=y_pred_lgbm.mean(axis=1)

# y_pred_ridge['avg']=y_pred_ridge.mean(axis=1)



#提出用ファイル

submission = pd.read_csv('../input/exam-for-students20200527/sample_submission.csv')

# submission.___ = (y_pred_lgbm['avg']+y_pred_ridge['avg'])/2

submission[target] = np.exp(y_pred_lgbm['avg'])+1

submission[target] = submission[target]#.round(-4)

submission.to_csv('submission.csv',index=False)