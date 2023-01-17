import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import lightgbm as lgbm

import xgboost as xgb

import datetime



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import preprocessing



pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 150)



import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
train.head(3)
train.info()
outcomes = train[['GameId','PlayId','Yards']].drop_duplicates()
def strtoseconds(txt):

    txt = txt.split(':')

    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60

    return ans



def strtofloat(x):

    try:

        return float(x)

    except:

        return -1



def map_weather(txt):

    ans = 1

    if pd.isna(txt):

        return 0

    if 'partly' in txt:

        ans*=0.5

    if 'climate controlled' in txt or 'indoor' in txt:

        return ans*3

    if 'sunny' in txt or 'sun' in txt:

        return ans*2

    if 'clear' in txt:

        return ans

    if 'cloudy' in txt:

        return -ans

    if 'rain' in txt or 'rainy' in txt:

        return -2*ans

    if 'snow' in txt:

        return -3*ans

    return 0



def OffensePersonnelSplit(x):

    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0, 'QB' : 0, 'RB' : 0, 'TE' : 0, 'WR' : 0}

    for xx in x.split(","):

        xxs = xx.split(" ")

        dic[xxs[-1]] = int(xxs[-2])

    return dic



def DefensePersonnelSplit(x):

    dic = {'DB' : 0, 'DL' : 0, 'LB' : 0, 'OL' : 0}

    for xx in x.split(","):

        xxs = xx.split(" ")

        dic[xxs[-1]] = int(xxs[-2])

    return dic



def orientation_to_cat(x):

    x = np.clip(x, 0, 360 - 1)

    try:

        return str(int(x/15))

    except:

        return "nan"

def preprocess(train):

    ## GameClock

    train['GameClock_sec'] = train['GameClock'].apply(strtoseconds)

    train["GameClock_minute"] = train["GameClock"].apply(lambda x : x.split(":")[0]).astype("object")



    ## Height

    train['PlayerHeight_dense'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))



    ## Time

    train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

    train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))



    train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

    train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))



    ## Age

    seconds_in_year = 60*60*24*365.25

    train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)

    train["PlayerAge_ob"] = train['PlayerAge'].astype(np.int).astype("object")



    ## WindSpeed

    train['WindSpeed_ob'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)

    train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)

    train['WindSpeed_ob'] = train['WindSpeed_ob'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)

    train['WindSpeed_dense'] = train['WindSpeed_ob'].apply(strtofloat)



    ## Weather

    train['GameWeather_process'] = train['GameWeather'].str.lower()

    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: "indoor" if not pd.isna(x) and "indoor" in x else x)

    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)

    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)

    train['GameWeather_process'] = train['GameWeather_process'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)

    train['GameWeather_dense'] = train['GameWeather_process'].apply(map_weather)



    ## Rusher

    train['IsRusher'] = (train['NflId'] == train['NflIdRusher'])

    train['IsRusher_ob'] = (train['NflId'] == train['NflIdRusher']).astype("object")

    temp = train[train["IsRusher"]][["Team", "PlayId"]].rename(columns={"Team":"RusherTeam"})

    train = train.merge(temp, on = "PlayId")

    train["IsRusherTeam"] = train["Team"] == train["RusherTeam"]



    ## dense -> categorical

    train["Quarter_ob"] = train["Quarter"].astype("object")

    train["Down_ob"] = train["Down"].astype("object")

    train["JerseyNumber_ob"] = train["JerseyNumber"].astype("object")

    train["YardLine_ob"] = train["YardLine"].astype("object")

  





    ## Orientation and Dir

    train["Orientation_ob"] = train["Orientation"].apply(lambda x : orientation_to_cat(x)).astype("object")

    train["Dir_ob"] = train["Dir"].apply(lambda x : orientation_to_cat(x)).astype("object")



    train["Orientation_sin"] = train["Orientation"].apply(lambda x : np.sin(x/360 * 2 * np.pi))

    train["Orientation_cos"] = train["Orientation"].apply(lambda x : np.cos(x/360 * 2 * np.pi))

    train["Dir_sin"] = train["Dir"].apply(lambda x : np.sin(x/360 * 2 * np.pi))

    train["Dir_cos"] = train["Dir"].apply(lambda x : np.cos(x/360 * 2 * np.pi))



    ## diff Score

    train["diffScoreBeforePlay"] = train["HomeScoreBeforePlay"] - train["VisitorScoreBeforePlay"]

    train["diffScoreBeforePlay_binary_ob"] = (train["HomeScoreBeforePlay"] > train["VisitorScoreBeforePlay"]).astype("object")



    ## Turf

    Turf = {'Field Turf':'Artificial', 'A-Turf Titan':'Artificial', 'Grass':'Natural', 'UBU Sports Speed S5-M':'Artificial', 'Artificial':'Artificial', 'DD GrassMaster':'Artificial', 'Natural Grass':'Natural', 'UBU Speed Series-S5-M':'Artificial', 'FieldTurf':'Artificial', 'FieldTurf 360':'Artificial', 'Natural grass':'Natural', 'grass':'Natural', 'Natural':'Natural', 'Artifical':'Artificial', 'FieldTurf360':'Artificial', 'Naturall Grass':'Natural', 'Field turf':'Artificial', 'SISGrass':'Artificial', 'Twenty-Four/Seven Turf':'Artificial', 'natural grass':'Natural'} 

    train['Turf'] = train['Turf'].map(Turf)



    ## OffensePersonnel

    temp = train["OffensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(lambda x : pd.Series(OffensePersonnelSplit(x)))

    temp.columns = ["Offense" + c for c in temp.columns]

    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]

    train = train.merge(temp, on = "PlayId")



    ## DefensePersonnel

    temp = train["DefensePersonnel"].iloc[np.arange(0, len(train), 22)].apply(lambda x : pd.Series(DefensePersonnelSplit(x)))

    temp.columns = ["Defense" + c for c in temp.columns]

    temp["PlayId"] = train["PlayId"].iloc[np.arange(0, len(train), 22)]

    train = train.merge(temp, on = "PlayId")



    ## sort

    train = train.sort_values(by = ['X']).sort_values(by = ['Dis']).sort_values(by=['PlayId', 'IsRusherTeam', 'IsRusher']).reset_index(drop = True)

    return train
train = preprocess(train)
train.head(5)
train.info()
# Thanks to : https://www.kaggle.com/aantonova/some-new-risk-and-clusters-features

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

categorical_columns = []

features = train.columns.values.tolist()

for col in features:

    if train[col].dtype in numerics: continue

    categorical_columns.append(col)

indexer = {}

for col in categorical_columns:

    if train[col].dtype in numerics: continue

    _, indexer[col] = pd.factorize(train[col])

    

for col in categorical_columns:

    if train[col].dtype in numerics: continue

    train[col] = indexer[col].get_indexer(train[col])
target = train['Yards']

del train['Yards']
train = train.fillna(-999)
train = reduce_mem_usage(train)
train.info()
X = train

z = target
#%% split training set to validation set

Xtrain, Xval, Ztrain, Zval = train_test_split(X, z, test_size=0.2, random_state=0)

train_set = lgbm.Dataset(Xtrain, Ztrain, silent=False)

valid_set = lgbm.Dataset(Xval, Zval, silent=False)
params = {

        'boosting_type':'gbdt',

        'objective': 'regression',

        'num_leaves': 31,

        'learning_rate': 0.05,

        'max_depth': -1,

        'subsample': 0.8,

        'bagging_fraction' : 1,

        'max_bin' : 5000 ,

        'bagging_freq': 20,

        'colsample_bytree': 0.6,

        'metric': 'rmse',

        'min_split_gain': 0.5,

        'min_child_weight': 1,

        'min_child_samples': 10,

        'scale_pos_weight':1,

        'zero_as_missing': True,

        'seed':0,        

    }



modelL = lgbm.train(params, train_set = train_set, num_boost_round=1000,

                   early_stopping_rounds=50,verbose_eval=10, valid_sets=valid_set)
fig =  plt.figure(figsize = (25,30))

axes = fig.add_subplot(111)

lgbm.plot_importance(modelL,ax = axes,height = 0.5)

plt.show();plt.close()
feature_score = pd.DataFrame(train.columns, columns = ['feature']) 

feature_score['score_lgb'] = modelL.feature_importance()
#%% split training set to validation set 

data_tr  = xgb.DMatrix(Xtrain, label=Ztrain)

data_cv  = xgb.DMatrix(Xval   , label=Zval)

evallist = [(data_tr, 'train'), (data_cv, 'valid')]
parms = {'max_depth':8, #maximum depth of a tree

         'objective':'reg:squarederror',

         'eta'      :0.3,

         'subsample':0.8,#SGD will use this percentage of data

         'lambda '  :4, #L2 regularization term,>1 more conservative 

         'colsample_bytree ':0.9,

         'colsample_bylevel':1,

         'min_child_weight': 10}

modelx = xgb.train(parms, data_tr, num_boost_round=200, evals = evallist,

                  early_stopping_rounds=30, maximize=False, 

                  verbose_eval=10)



print('score = %1.5f, n_boost_round =%d.'%(modelx.best_score,modelx.best_iteration))
fig =  plt.figure(figsize = (15,30))

axes = fig.add_subplot(111)

xgb.plot_importance(modelx,ax = axes,height = 0.5)

plt.show();plt.close()
feature_score['score_xgb'] = feature_score['feature'].map(modelx.get_score(importance_type='weight'))

feature_score
# Standardization for regression model

train = pd.DataFrame(

    preprocessing.MinMaxScaler().fit_transform(train),

    columns=train.columns,

    index=train.index

)
# Linear Regression



linreg = LinearRegression()

linreg.fit(train, target)

coeff_linreg = pd.DataFrame(train.columns.delete(0))

coeff_linreg.columns = ['feature']

coeff_linreg["score_linreg"] = pd.Series(linreg.coef_)

coeff_linreg.sort_values(by='score_linreg', ascending=False)
# the level of importance of features is not associated with the sign

coeff_linreg["score_linreg"] = coeff_linreg["score_linreg"].abs()



feature_score = pd.merge(feature_score, coeff_linreg, on='feature')

feature_score = feature_score.fillna(0)

feature_score = feature_score.set_index('feature')

feature_score
#Thanks to https://www.kaggle.com/nanomathias/feature-engineering-importance-testing

# MinMax scale all importances

feature_score = pd.DataFrame(

    preprocessing.MinMaxScaler().fit_transform(feature_score),

    columns=feature_score.columns,

    index=feature_score.index

)



# Create mean column

feature_score['mean'] = feature_score.mean(axis=1)



# Create total column with different weights

feature_score['weighted average - 0.5/0.35/0.15'] = 0.5*feature_score['score_lgb'] + 0.35*feature_score['score_xgb'] + 0.15*feature_score['score_linreg']



# Plot the feature importances

feature_score.sort_values('weighted average - 0.5/0.35/0.15', ascending=False).plot(kind='bar', figsize=(20, 15))
feature_score.sort_values('weighted average - 0.5/0.35/0.15', ascending=False)