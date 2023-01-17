# 가상 GPU

!nvidia-smi
# package

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import json

import ast

from collections import Counter

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import lightgbm as lgb

import time

from datetime import datetime

import eli5

import seaborn as sns

from scipy.stats import skew, boxcox

import xgboost as xgb

from pandas_summary import DataFrameSummary
# method

def date_features(df):

    df[['release_month','release_day','release_year']]=df['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)

    # 연도 끝 두자리수만 있기 때문에 앞에 19/20 붙이기

    df.loc[ (train['release_year'] <= 19) & (df['release_year'] < 100), "release_year"] += 2000

    df.loc[ (train['release_year'] > 19) & (df['release_year'] < 100), "release_year"] += 1900

    return df



def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df
# Loading dataset

submission_path = "../input/sample_submission.csv"

train_path = "../input/train.csv"

test_path = "../input/test.csv"



train = pd.read_csv(train_path)

test = pd.read_csv(test_path)

submission = pd.read_csv(submission_path)

print( "train dataset:", train.shape,"\n","test dataset: ",test.shape,"\n","sample_submission dataset:", submission.shape)
# train 데이터 문제점 개선

train.loc[train['id'] == 16,'revenue'] = 192864 # Skinning

train.loc[train['id'] == 90,'budget'] = 30000000 # Sommersby

train.loc[train['id'] == 118,'budget'] = 60000000 # Wild Hogs

train.loc[train['id'] == 149,'budget'] = 18000000 # Beethoven

train.loc[train['id'] == 313,'revenue'] = 12000000 # The Cookout

train.loc[train['id'] == 451,'revenue'] = 12000000 # Chasing Liberty

train.loc[train['id'] == 464,'budget'] = 20000000 # Parenthood

train.loc[train['id'] == 470,'budget'] = 13000000 # The Karate Kid, Part II

train.loc[train['id'] == 513,'budget'] = 930000 # From Prada to Nada

train.loc[train['id'] == 797,'budget'] = 8000000 # Welcome to Dongmakgol

train.loc[train['id'] == 819,'budget'] = 90000000 # Alvin and the Chipmunks: The Road Chip

train.loc[train['id'] == 850,'budget'] = 90000000 # Modern Times

train.loc[train['id'] == 1007,'budget'] = 2 # Zyzzyx Road

train.loc[train['id'] == 1112,'budget'] = 7500000 # An Officer and a Gentleman

train.loc[train['id'] == 1131,'budget'] = 4300000 # Smokey and the Bandit

train.loc[train['id'] == 1359,'budget'] = 10000000 # Stir Crazy

train.loc[train['id'] == 1542,'budget'] = 1 # All at Once

train.loc[train['id'] == 1570,'budget'] = 15800000 # Crocodile Dundee II

train.loc[train['id'] == 1571,'budget'] = 4000000 # Lady and the Tramp

train.loc[train['id'] == 1714,'budget'] = 46000000 # The Recruit

train.loc[train['id'] == 1721,'budget'] = 17500000 # Cocoon

train.loc[train['id'] == 1865,'revenue'] = 25000000 # Scooby-Doo 2: Monsters Unleashed

train.loc[train['id'] == 1885,'budget'] = 12 # In the Cut

train.loc[train['id'] == 2091,'budget'] = 10 # Deadfall

train.loc[train['id'] == 2268,'budget'] = 17500000 # Madea Goes to Jail budget

train.loc[train['id'] == 2491,'budget'] = 6 # Never Talk to Strangers

train.loc[train['id'] == 2602,'budget'] = 31000000 # Mr. Holland's Opus

train.loc[train['id'] == 2612,'budget'] = 15000000 # Field of Dreams

train.loc[train['id'] == 2696,'budget'] = 10000000 # Nurse 3-D

train.loc[train['id'] == 2801,'budget'] = 10000000 # Fracture

train.loc[train['id'] == 335,'budget'] = 2

train.loc[train['id'] == 348,'budget'] = 12

train.loc[train['id'] == 470,'budget'] = 13000000

train.loc[train['id'] == 513,'budget'] = 1100000

train.loc[train['id'] == 640,'budget'] = 6

train.loc[train['id'] == 696,'budget'] = 1

train.loc[train['id'] == 797,'budget'] = 8000000

train.loc[train['id'] == 850,'budget'] = 1500000

train.loc[train['id'] == 1199,'budget'] = 5

train.loc[train['id'] == 1282,'budget'] = 9 # Death at a Funeral

train.loc[train['id'] == 1347,'budget'] = 1

train.loc[train['id'] == 1755,'budget'] = 2

train.loc[train['id'] == 1801,'budget'] = 5

train.loc[train['id'] == 1918,'budget'] = 592

train.loc[train['id'] == 2033,'budget'] = 4

train.loc[train['id'] == 2118,'budget'] = 344

train.loc[train['id'] == 2252,'budget'] = 130

train.loc[train['id'] == 2256,'budget'] = 1

train.loc[train['id'] == 2696,'budget'] = 10000000



# test 데이터 문제점 개선



test

test.loc[test['id'] == 6733,'budget'] = 5000000

test.loc[test['id'] == 3889,'budget'] = 15000000

test.loc[test['id'] == 6683,'budget'] = 50000000

test.loc[test['id'] == 5704,'budget'] = 4300000

test.loc[test['id'] == 6109,'budget'] = 281756

test.loc[test['id'] == 7242,'budget'] = 10000000

test.loc[test['id'] == 7021,'budget'] = 17540562 # Two Is a Family

test.loc[test['id'] == 5591,'budget'] = 4000000 # The Orphanage

test.loc[test['id'] == 4282,'budget'] = 20000000 # Big Top Pee-wee

test.loc[test['id'] == 3033,'budget'] = 250

test.loc[test['id'] == 3051,'budget'] = 50

test.loc[test['id'] == 3084,'budget'] = 337

test.loc[test['id'] == 3224,'budget'] = 4

test.loc[test['id'] == 3594,'budget'] = 25

test.loc[test['id'] == 3619,'budget'] = 500

test.loc[test['id'] == 3831,'budget'] = 3

test.loc[test['id'] == 3935,'budget'] = 500

test.loc[test['id'] == 4049,'budget'] = 995946

test.loc[test['id'] == 4424,'budget'] = 3

test.loc[test['id'] == 4460,'budget'] = 8

test.loc[test['id'] == 4555,'budget'] = 1200000

test.loc[test['id'] == 4624,'budget'] = 30

test.loc[test['id'] == 4645,'budget'] = 500

test.loc[test['id'] == 4709,'budget'] = 450

test.loc[test['id'] == 4839,'budget'] = 7

test.loc[test['id'] == 3125,'budget'] = 25

test.loc[test['id'] == 3142,'budget'] = 1

test.loc[test['id'] == 3201,'budget'] = 450

test.loc[test['id'] == 3222,'budget'] = 6

test.loc[test['id'] == 3545,'budget'] = 38

test.loc[test['id'] == 3670,'budget'] = 18

test.loc[test['id'] == 3792,'budget'] = 19

test.loc[test['id'] == 3881,'budget'] = 7

test.loc[test['id'] == 3969,'budget'] = 400

test.loc[test['id'] == 4196,'budget'] = 6

test.loc[test['id'] == 4221,'budget'] = 11

test.loc[test['id'] == 4222,'budget'] = 500

test.loc[test['id'] == 4285,'budget'] = 11

test.loc[test['id'] == 4319,'budget'] = 1

test.loc[test['id'] == 4639,'budget'] = 10

test.loc[test['id'] == 4719,'budget'] = 45

test.loc[test['id'] == 4822,'budget'] = 22

test.loc[test['id'] == 4829,'budget'] = 20

test.loc[test['id'] == 4969,'budget'] = 20

test.loc[test['id'] == 5021,'budget'] = 40

test.loc[test['id'] == 5035,'budget'] = 1

test.loc[test['id'] == 5063,'budget'] = 14

test.loc[test['id'] == 5119,'budget'] = 2

test.loc[test['id'] == 5214,'budget'] = 30

test.loc[test['id'] == 5221,'budget'] = 50

test.loc[test['id'] == 4903,'budget'] = 15

test.loc[test['id'] == 4983,'budget'] = 3

test.loc[test['id'] == 5102,'budget'] = 28

test.loc[test['id'] == 5217,'budget'] = 75

test.loc[test['id'] == 5224,'budget'] = 3

test.loc[test['id'] == 5469,'budget'] = 20

test.loc[test['id'] == 5840,'budget'] = 1

test.loc[test['id'] == 5960,'budget'] = 30

test.loc[test['id'] == 6506,'budget'] = 11

test.loc[test['id'] == 6553,'budget'] = 280

test.loc[test['id'] == 6561,'budget'] = 7

test.loc[test['id'] == 6582,'budget'] = 218

test.loc[test['id'] == 6638,'budget'] = 5

test.loc[test['id'] == 6749,'budget'] = 8

test.loc[test['id'] == 6759,'budget'] = 50

test.loc[test['id'] == 6856,'budget'] = 10

test.loc[test['id'] == 6858,'budget'] = 100

test.loc[test['id'] == 6876,'budget'] = 250

test.loc[test['id'] == 6972,'budget'] = 1

test.loc[test['id'] == 7079,'budget'] = 8000000

test.loc[test['id'] == 7150,'budget'] = 118

test.loc[test['id'] == 6506,'budget'] = 118

test.loc[test['id'] == 7225,'budget'] = 6

test.loc[test['id'] == 7231,'budget'] = 85

test.loc[test['id'] == 5222,'budget'] = 5

test.loc[test['id'] == 5322,'budget'] = 90

test.loc[test['id'] == 5350,'budget'] = 70

test.loc[test['id'] == 5378,'budget'] = 10

test.loc[test['id'] == 5545,'budget'] = 80

test.loc[test['id'] == 5810,'budget'] = 8

test.loc[test['id'] == 5926,'budget'] = 300

test.loc[test['id'] == 5927,'budget'] = 4

test.loc[test['id'] == 5986,'budget'] = 1

test.loc[test['id'] == 6053,'budget'] = 20

test.loc[test['id'] == 6104,'budget'] = 1

test.loc[test['id'] == 6130,'budget'] = 30

test.loc[test['id'] == 6301,'budget'] = 150

test.loc[test['id'] == 6276,'budget'] = 100

test.loc[test['id'] == 6473,'budget'] = 100

test.loc[test['id'] == 6842,'budget'] = 30
# date_features

train = date_features(train)

test = date_features(test)



# text_to_dict

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



train = text_to_dict(train)

test = text_to_dict(test)
train['num_genres'] = train['genres'].apply(lambda x: len(x) if x != {} else 0)



test['num_genres'] = test['genres'].apply(lambda x: len(x) if x != {} else 0)
train['num_production_companies'] = train['production_companies'].apply(lambda x: len(x) if x != {} else 0)



test['num_production_companies'] = test['production_companies'].apply(lambda x: len(x) if x != {} else 0)
train['num_production_countries'] = train['production_countries'].apply(lambda x: len(x) if x != {} else 0)



test['num_production_countries'] = test['production_countries'].apply(lambda x: len(x) if x != {} else 0)
train['num_spoken_languages'] = train['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)



test['num_spoken_languages'] = test['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)
train['num_Keywords'] = train['Keywords'].apply(lambda x: len(x) if x != {} else 0)



test['num_Keywords'] = test['Keywords'].apply(lambda x: len(x) if x != {} else 0)
train['has_collection'] = train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)



test['has_collection'] = test['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)
train['has_homepage'] = 1

train.loc[pd.isnull(train['homepage']) ,"has_homepage"] = 0

test['has_homepage'] = 1

test.loc[pd.isnull(test['homepage']) ,"has_homepage"] = 0
train['isTaglineNA'] = 0

train.loc[pd.isnull(train['tagline']) ,"isTaglineNA"] = 1

test['isTaglineNA'] = 0

test.loc[pd.isnull(test['tagline']) ,"isTaglineNA"] = 1
train['isOriginalLanguageEng'] = 0

test['isOriginalLanguageEng'] = 0



train.loc[train['original_language'] == "en" ,"isOriginalLanguageEng"] = 1

test.loc[test['original_language'] == "en" ,"isOriginalLanguageEng"] = 1
train['all_spoken_languages'] = train['spoken_languages'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')

test['all_spoken_languages'] = test['spoken_languages'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')



train['isSpokenLanguageEng'] = 0

train.loc[train['all_spoken_languages'] == "English" ,"isSpokenLanguageEng"] = 1

test['isSpokenLanguageEng'] = 0

test.loc[test['all_spoken_languages'] == "English" ,"isSpokenLanguageEng"] = 1
train['log_budget']=np.log1p(train['budget'] + 1)

test['log_budget']=np.log1p(test['budget'] + 1)
train['isTitleDifferent'] = 1

train.loc[ train['original_title'] == train['title'] ,"isTitleDifferent"] = 0

test['isTitleDifferent'] = 1

test.loc[ test['original_title'] == test['title'] ,"isTitleDifferent"] = 0
train['budget_year_ratio'] = train['budget'] / (train['release_year'] * train['release_year']) 

test['budget_year_ratio'] = test['budget'] / (test['release_year'] * test['release_year'])
# cast

train['all_cast'] = train['cast'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')

test['all_cast'] = test['cast'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')



list_of_cast_names_tr = list(train['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

top_cast_names_tr = [m[0] for m in Counter([i for j in list_of_cast_names_tr for i in j]).most_common(10)]

list_of_cast_names_tt = list(test['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

top_cast_names_tt = [m[0] for m in Counter([i for j in list_of_cast_names_tt for i in j]).most_common(10)]





for g in top_cast_names_tr:

    train['cast_name_' + g] = train['all_cast'].apply(lambda x: 1 if g in x else 0)



for g in top_cast_names_tt:

    test['cast_name_' + g] = test['all_cast'].apply(lambda x: 1 if g in x else 0)



num_top_cast_tr = []

num_top_cast_tt = []



for i in range(len(train.index)):

    names = set(train.all_cast[i])

    topnames = set(top_cast_names_tr)

    num_top_cast_tr.append(len(names&topnames))

    

for i in range(len(test.index)):

    names = set(test.all_cast[i])

    topnames = set(top_cast_names_tt)

    num_top_cast_tt.append(len(names&topnames))



train["num_top_cast"] = num_top_cast_tr

test["num_top_cast"] = num_top_cast_tt



# crew



train['all_crew'] = train['crew'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')

test['all_crew'] = test['crew'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')



list_of_crew_names_tr = list(train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

top_crew_names_tr = [m[0] for m in Counter([i for j in list_of_crew_names_tr for i in j]).most_common(10)]

list_of_crew_names_tt = list(test['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

top_crew_names_tt = [m[0] for m in Counter([i for j in list_of_crew_names_tt for i in j]).most_common(10)]





for g in top_crew_names_tr:

    train['crew_name_' + g] = train['all_crew'].apply(lambda x: 1 if g in x else 0)



for g in top_crew_names_tt:

    test['crew_name_' + g] = test['all_crew'].apply(lambda x: 1 if g in x else 0)



num_top_crew_tr = []

num_top_crew_tt = []



for i in range(len(train.index)):

    names = set(train.all_crew[i])

    topnames = set(top_crew_names_tr)

    num_top_crew_tr.append(len(names&topnames))

    

for i in range(len(test.index)):

    names = set(test.all_crew[i])

    topnames = set(top_crew_names_tt)

    num_top_crew_tt.append(len(names&topnames))



train["num_top_crew"] = num_top_crew_tr

test["num_top_crew"] = num_top_crew_tt
# category



cat_features = [ 'has_homepage',

                 'isTaglineNA',

                 'isTitleDifferent', 

                 'has_collection', 

                 'isOriginalLanguageEng',

                 'isSpokenLanguageEng']

 

for i in enumerate (cat_features) :

    ca = i[1]

    train[ca] = train[ca].astype('category')

    test[ca] = test[ca].astype('category')
# Formating for modeling



used_features = ['release_year',

                 'num_genres', 

                 'num_production_companies', 

                 'num_production_countries', 

                 'runtime',  

                 'num_spoken_languages',

                 'num_Keywords', 

                 'has_homepage',

                 'isTaglineNA',

                 'isTitleDifferent', 

                 'log_budget', 

                 'has_collection', 

                 'isOriginalLanguageEng',

                 'isSpokenLanguageEng',

                 'budget_year_ratio',

                 "num_top_crew",

                 "num_top_cast"]





X = train[used_features]

y = np.log1p(train['revenue'] + 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# parameter



# params = {'num_leaves': 30,

# #          'min_data_in_leaf': 20,

# #          'objective': 'regression',

#          'max_depth': 5,

#          'learning_rate': 0.01,

#          "boosting": "gbdt"}

# #          "feature_fraction": 0.9,

# #          "bagging_freq": 1,

# #          "bagging_fraction": 0.9,

# #          "bagging_seed": 11,

# #          "metric": 'rmse',

# #          "lambda_l1": 0.2,

#          "verbosity": -1}
lgb_model = lgb.LGBMRegressor(n_estimators = 10000, nthread = 4, n_jobs = -1)

lgb_model.fit(X_train, y_train, 

        eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=200)



print("훈련 점수: {:.2f}".format(lgb_model.score(X_train, y_train)))

print("테스트 점수: {:.2f}".format(lgb_model.score(X_test, y_test)))



eli5.show_weights(lgb_model, feature_filter=lambda x: x != '<BIAS>')
xgb_model = xgb.XGBRegressor()

xgb_model.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=200)



print("훈련 점수: {:.2f}".format(xgb_model.score(X_train, y_train)))

print("테스트 점수: {:.2f}".format(xgb_model.score(X_test, y_test)))



eli5.show_weights(xgb_model, feature_filter=lambda x: x != '<BIAS>')
y_pred = lgb_model.predict(test[used_features])

submission['revenue'] = y_pred

submission.to_csv('2nd_submission.csv', index=False)