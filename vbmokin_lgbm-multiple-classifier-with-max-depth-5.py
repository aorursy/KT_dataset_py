import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import datetime

from kaggle.competitions import nflrush

import tqdm

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler, LabelEncoder

import keras



from tqdm import tqdm_notebook

import warnings

warnings.filterwarnings('ignore')



sns.set_style('darkgrid')

mpl.rcParams['figure.figsize'] = [15,10]
train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})

print(train.shape)

train.head()
#https://www.kaggle.com/rooshroosh/fork-of-neural-networks-different-architecture

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
%%time

train = preprocess(train)
## DisplayName remove Outlier

v = train["DisplayName"].value_counts()

missing_values = list(v[v < 5].index)

train["DisplayName"] = train["DisplayName"].where(~train["DisplayName"].isin(missing_values), "nan")



## PlayerCollegeName remove Outlier

v = train["PlayerCollegeName"].value_counts()

missing_values = list(v[v < 10].index)

train["PlayerCollegeName"] = train["PlayerCollegeName"].where(~train["PlayerCollegeName"].isin(missing_values), "nan")
pd.to_pickle(train, "train.pkl")
def drop(train):

    drop_cols = ["GameId", "GameWeather", "NflId", "Season", "NflIdRusher"] 

    drop_cols += ['TimeHandoff', 'TimeSnap', 'PlayerBirthDate']

    drop_cols += ["Orientation", "Dir", 'WindSpeed', "GameClock"]

    train = train.drop(drop_cols, axis = 1)

    return train
train = drop(train)
un_use_features = ['G_HomeTeamAbbr',

 'G_PossessionTeam',

 'G_RusherTeam',

 'G_StadiumType',

 'G_diffScoreBeforePlay_binary_ob',

 'P_DefenseDB',

 'P_DefenseDL',

 'P_DefenseLB',

 'P_DefenseOL',

 'P_GameWeather_dense',

 'P_IsRusher',

 'P_IsRusherTeam',

 'P_OffenseDB',

 'P_OffenseDL',

 'P_OffenseLB',

 'P_OffenseOL',

 'P_OffenseQB',

 'P_OffenseTE',

 'P_OffenseWR',

 'P_PlayerAge_ob',

 'P_Quarter',

 'P_Week']



un_use_features += ['G_Down_ob',

 'G_FieldPosition',

 'G_OffenseRB',

 'G_TimeDelta',

 'G_VisitorTeamAbbr',

 'G_WindDirection',

 'P_GameClock_sec',

 'P_HomeScoreBeforePlay',

 'P_Humidity',

 'P_Orientation_ob',

 'P_PlayerHeight',

 'P_Temperature',

 'P_VisitorScoreBeforePlay',

 'P_WindSpeed_dense',

 'P_diffScoreBeforePlay']



## delete prefix

un_use_features = [c[2:] for c in un_use_features]

train = train.drop(un_use_features, axis = 1)
cat_features = []

dense_features = []

for col in train.columns:

    if train[col].dtype =='object':

        cat_features.append(col)

        print("*cat*", col, len(train[col].unique()))

    else:

        dense_features.append(col)

        print("!dense!", col, len(train[col].unique()))

dense_features.remove("PlayId")

dense_features.remove("Yards")
train_cat = train[cat_features]

categories = []

most_appear_each_categories = {}

for col in tqdm_notebook(train_cat.columns):

    train_cat.loc[:,col] = train_cat[col].fillna("nan")

    train_cat.loc[:,col] = col + "__" + train_cat[col].astype(str)

    most_appear_each_categories[col] = list(train_cat[col].value_counts().index)[0]

    categories.append(train_cat[col].unique())

categories = np.hstack(categories)

print(len(categories))
le = LabelEncoder()

le.fit(categories)

for col in tqdm_notebook(train_cat.columns):

    train_cat.loc[:, col] = le.transform(train_cat[col])

num_classes = len(le.classes_)
train_dense = train[dense_features]

sss = {}

medians = {}

for col in tqdm_notebook(train_dense.columns):

    print(col)

    medians[col] = np.nanmedian(train_dense[col])

    train_dense.loc[:, col] = train_dense[col].fillna(medians[col])

    ss = StandardScaler()

    train_dense.loc[:, col] = ss.fit_transform(train_dense[col].values[:,None])

    sss[col] = ss
eps = 1e-8

## dense features for play

dense_game_features = train_dense.columns[train_dense[:22].std() <= eps]

## dense features for each player

dense_player_features = train_dense.columns[train_dense[:22].std() > eps]

## categorical features for play

cat_game_features = train_cat.columns[train_cat[:22].std() <= eps]

## categorical features for each player

cat_player_features = train_cat.columns[train_cat[:22].std() > eps]
dense_game_feature_names = ["G_" + cc for cc in dense_game_features]

dense_player_feature_names = list(np.hstack([["P_" + c for c in dense_player_features] for k in range(22)]))

cat_game_feature_names = ["G_" + cc for cc in cat_game_features]

cat_player_feature_names = list(np.hstack([["P_" + c for c in cat_player_features] for k in range(22)]))
train_dense_game = train_dense[dense_game_features].iloc[np.arange(0, len(train), 22)].reset_index(drop = True).values

train_dense_players = [train_dense[dense_player_features].iloc[np.arange(k, len(train), 22)].reset_index(drop = True) for k in range(22)]

train_dense_players = np.stack([t.values for t in train_dense_players]).transpose(1, 0, 2)

train_cat_game = train_cat[cat_game_features].iloc[np.arange(0, len(train), 22)].reset_index(drop = True).values

train_cat_players = [train_cat[cat_player_features].iloc[np.arange(k, len(train), 22)].reset_index(drop = True) for k in range(22)]

train_cat_players = np.stack([t.values for t in train_cat_players]).transpose(1, 0, 2)
def return_step(x):

    temp = np.zeros(199)

    temp[x + 99:] = 1

    return temp



train_y_raw = train["Yards"].iloc[np.arange(0, len(train), 22)].reset_index(drop = True)

train_y = np.vstack(train_y_raw.apply(return_step).values)
train_dense_game.shape, train_dense_players.shape, train_cat_game.shape, train_cat_players.shape, train_y.shape
## concat all features

train_dense_players = np.reshape(train_dense_players, (len(train_dense_players), -1))

train_dense = np.hstack([train_dense_players, train_dense_game])



train_cat_players = np.reshape(train_cat_players, (len(train_cat_players), -1))

train_cat = np.hstack([train_cat_players, train_cat_game])



train_x = np.hstack([train_dense, train_cat])
from lightgbm import LGBMClassifier

class MultiLGBMClassifier():

    def __init__(self, resolution, params):

        ## smoothing size

        self.resolution = resolution

        ## initiarize models

        self.models = [LGBMClassifier(**params) for _ in range(resolution)]

        

    def fit(self, x, y):

        self.classes_list = []

        for k in tqdm_notebook(range(self.resolution)):

            ## train each model

            self.models[k].fit(x, (y + k) // self.resolution)

            ## (0,1,2,3,4,5,6,7,8,9) -> (0,0,0,0,0,1,1,1,1,1) -> (0,5)

            classes = np.sort(list(set((y + k) // self.resolution))) * self.resolution - k

            classes = np.append(classes, 999)

            self.classes_list.append(classes)

            

    def predict(self, x):

        pred199_list = []

        for k in range(self.resolution):

            preds = self.models[k].predict_proba(x)

            classes = self.classes_list[k]

            pred199s = self.get_pred199(preds, classes)

            pred199_list.append(pred199s)

        self.pred199_list = pred199_list

        pred199_ens = np.mean(np.stack(pred199_list), axis = 0)

        return pred199_ens

    

    def _get_pred199(self, p, classes):

        ## categorical prediction -> predicted distribution whose length is 199

        pred199 = np.zeros(199)

        for k in range(len(p)):

            pred199[classes[k] + 99 : classes[k+1] + 99] = p[k]

        return pred199



    def get_pred199(self, preds, classes):

        pred199s = []

        for p in preds:

            pred199 = np.cumsum(self._get_pred199(p, classes))

            pred199 = pred199/np.max(pred199)

            pred199s.append(pred199)

        return np.vstack(pred199s)
params = {'num_leaves': 35, 'max_depth': 5, # 2**5 = 32 - Let's set num_leaves=35

 'subsample': 0.4, 'min_child_samples': 10,

 'learning_rate': 0.01,

 'num_iterations': 500, 'random_state': 12}
from sklearn.model_selection import train_test_split, KFold

losses = []

models = []

for k in range(1):

    kfold = KFold(5, random_state = 42 + k, shuffle = True)

    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(train_y)):

        print("-----------")

        print("-----------")

        model = MultiLGBMClassifier(resolution = 5, params = params)

        model.fit(train_x[tr_inds], train_y_raw.values[tr_inds])

        preds = model.predict(train_x[val_inds])

        loss = np.mean((train_y[val_inds] - preds) ** 2)

        models.append(model)

        print(k_fold, loss)

        losses.append(loss)

print("-------")

print(losses)

print(np.mean(losses))
print(losses)

print(np.mean(losses))
feature_importances = 0

num_model = 0

for model in models:

    for m in model.models:

        feature_importances += m.booster_.feature_importance("gain")

        num_model += 1



feature_importances /= num_model
feature_names = dense_player_feature_names + dense_game_feature_names + cat_game_feature_names + cat_player_feature_names

feature_importance_df = pd.DataFrame(np.vstack([feature_importances, feature_names]).T, columns = ["importance", "name"])

feature_importance_df["importance"] = feature_importance_df["importance"].astype(np.float32)

feature_importance_df = feature_importance_df.groupby("name").agg("mean").reset_index()
plt.figure(figsize = (8, 18))

sns.barplot(data = feature_importance_df.sort_values(by = "importance", ascending = False).head(50), x = "importance", y = "name")

plt.show()
plt.figure(figsize = (8, 18))

sns.barplot(data = feature_importance_df.sort_values(by = "importance", ascending = False).tail(50), x = "importance", y = "name")

plt.show()
## bad features

list(feature_importance_df[feature_importance_df["importance"] < np.quantile(feature_importance_df["importance"], 0.3)]["name"])
def make_pred(test, sample, env, model):

    test = preprocess(test)

    test = drop(test)

    test = test.drop(un_use_features, axis = 1)

    

    ### categorical

    test_cat = test[cat_features]

    for col in (test_cat.columns):

        test_cat.loc[:,col] = test_cat[col].fillna("nan")

        test_cat.loc[:,col] = col + "__" + test_cat[col].astype(str)

        isnan = ~test_cat.loc[:,col].isin(categories)

        if np.sum(isnan) > 0:

            if not ((col + "__nan") in categories):

                test_cat.loc[isnan,col] = most_appear_each_categories[col]

            else:

                test_cat.loc[isnan,col] = col + "__nan"

    for col in (test_cat.columns):

        test_cat.loc[:, col] = le.transform(test_cat[col])



    ### dense

    test_dense = test[dense_features]

    for col in (test_dense.columns):

        test_dense.loc[:, col] = test_dense[col].fillna(medians[col])

        test_dense.loc[:, col] = sss[col].transform(test_dense[col].values[:,None])



    ### divide

    test_dense_players = [test_dense[dense_player_features].iloc[np.arange(k, len(test), 22)].reset_index(drop = True) for k in range(22)]

    test_dense_players = np.stack([t.values for t in test_dense_players]).transpose(1,0, 2)



    test_dense_game = test_dense[dense_game_features].iloc[np.arange(0, len(test), 22)].reset_index(drop = True).values

    

    test_cat_players = [test_cat[cat_player_features].iloc[np.arange(k, len(test), 22)].reset_index(drop = True) for k in range(22)]

    test_cat_players = np.stack([t.values for t in test_cat_players]).transpose(1,0, 2)



    test_cat_game = test_cat[cat_game_features].iloc[np.arange(0, len(test), 22)].reset_index(drop = True).values



    test_dense_players = np.reshape(test_dense_players, (len(test_dense_players), -1))

    test_dense = np.hstack([test_dense_players, test_dense_game])

    test_cat_players = np.reshape(test_cat_players, (len(test_cat_players), -1))

    test_cat = np.hstack([test_cat_players, test_cat_game])

    test_x = np.hstack([test_dense, test_cat])



    test_inp = test_x

    

    ## pred

    pred = 0

    for model in models:

        _pred = model.predict(test_inp)

        pred += _pred

    pred /= len(models)

    pred = np.clip(pred, 0, 1)

    env.predict(pd.DataFrame(data=pred,columns=sample.columns))

    return pred
env = nflrush.make_env()

preds = []

for test, sample in tqdm_notebook(env.iter_test()):

    pred = make_pred(test, sample, env, models)

    preds.append(pred)

env.write_submission_file()
preds = np.vstack(preds)

## check whether prediction is submittable

print(np.mean(np.diff(preds, axis = 1) >= 0) == 1.0)

print(np.mean(preds > 1) == 0)
print(losses)

print(np.mean(losses))