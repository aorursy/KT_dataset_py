# package

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import json

import ast

plt.style.use('ggplot')

from sklearn.model_selection import train_test_split

import lightgbm as lgb

import time

from datetime import datetime

import eli5
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
submission_path = "../input/sample_submission.csv"

train_path = "../input/train.csv"

test_path = "../input/test.csv"



train = pd.read_csv(train_path)

submission = pd.read_csv(submission_path)

test = pd.read_csv(test_path)
train = date_features(train)

test = date_features(test)
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',

                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']



train = text_to_dict(train)

test = text_to_dict(test)
train['collection_name'] = train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)

train['has_collection'] = train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)



test['collection_name'] = test['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)

test['has_collection'] = test['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)
train['num_genres'] = train['genres'].apply(lambda x: len(x) if x != {} else 0)

train['all_genres'] = train['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')



test['num_genres'] = test['genres'].apply(lambda x: len(x) if x != {} else 0)

test['all_genres'] = test['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
train['num_production_companies'] = train['production_companies'].apply(lambda x: len(x) if x != {} else 0)

train['all_production_companies'] = train['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')



test['num_production_companies'] = test['production_companies'].apply(lambda x: len(x) if x != {} else 0)

test['all_production_companies'] = test['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
train['num_production_countries'] = train['production_countries'].apply(lambda x: len(x) if x != {} else 0)

train['all_production_countries'] = train['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')



test['num_production_countries'] = test['production_countries'].apply(lambda x: len(x) if x != {} else 0)

test['all_production_countries'] = test['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
# list_of_cast_names = list(train['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

train['num_cast'] = train['cast'].apply(lambda x: len(x) if x != {} else 0)

train['all_cast'] = train['cast'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')



test['num_cast'] = test['cast'].apply(lambda x: len(x) if x != {} else 0)

test['all_cast'] = test['cast'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
# list_of_crew_names = list(train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

train['num_crew'] = train['crew'].apply(lambda x: len(x) if x != {} else 0)

train['all_crew'] = train['crew'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')



test['num_crew'] = test['crew'].apply(lambda x: len(x) if x != {} else 0)

test['all_crew'] = test['crew'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
# list_of_spokenlanguage_names = list(train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

train['num_spoken_languages'] = train['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

train['all_spoken_languages'] = train['spoken_languages'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')



test['num_spoken_languages'] = test['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

test['all_spoken_languages'] = test['spoken_languages'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
# list_of_Keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

train['num_Keywords'] = train['Keywords'].apply(lambda x: len(x) if x != {} else 0)

train['all_Keywords'] = train['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

test['num_Keywords'] = test['Keywords'].apply(lambda x: len(x) if x != {} else 0)

test['all_Keywords'] = test['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
# category

cat_features = ['all_production_companies',

'all_production_countries',

'all_genres',                

'all_cast',

'all_crew',

'all_spoken_languages',

'all_Keywords',

'collection_name']

 

for i in enumerate (cat_features) :

    ca = i[1]

    train[ca] = train[ca].astype('category')

    test[ca] = test[ca].astype('category')
train['has_homepage'] = 1

train.loc[pd.isnull(train['homepage']) ,"has_homepage"] = 0

test['has_homepage'] = 1

test.loc[pd.isnull(test['homepage']) ,"has_homepage"] = 0
train['isTaglineNA'] = 0

train.loc[pd.isnull(train['tagline']) ,"isTaglineNA"] = 1

test['isTaglineNA'] = 0

test.loc[pd.isnull(test['tagline']) ,"isTaglineNA"] = 1
train['isTitleDifferent'] = 1

train.loc[ train['original_title'] == train['title'] ,"isTitleDifferent"] = 0

test['isTitleDifferent'] = 1

test.loc[ test['original_title'] == test['title'] ,"isTitleDifferent"] = 0
train['isOriginalLanguageEng'] = 0

test['isOriginalLanguageEng'] = 0



train.loc[train['original_language'] == "en" ,"isOriginalLanguageEng"] = 1

test.loc[test['original_language'] == "en" ,"isOriginalLanguageEng"] = 1
train['isSpokenLanguageEng'] = 0

train.loc[train['all_spoken_languages'] == "English" ,"isSpokenLanguageEng"] = 1

test['isSpokenLanguageEng'] = 0

test.loc[test['all_spoken_languages'] == "English" ,"isSpokenLanguageEng"] = 1
# Formating for modeling



used_features = ['release_year', 'num_genres', 'all_genres',

       'num_production_companies', 'all_production_companies',

       'num_production_countries', 'all_production_countries', 'num_cast',

       'all_cast', 'num_crew', 'all_crew', 'num_spoken_languages',

       'all_spoken_languages', 'num_Keywords', 'all_Keywords', 'has_homepage',

       'isTaglineNA', 'isTitleDifferent', 'budget', 'runtime', 

                 'isOriginalLanguageEng', 'isSpokenLanguageEng', 'has_collection']



X = train[used_features]

y = train['revenue']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lgb_model = lgb.LGBMRegressor(n_estimators = 10000, nthread = 4, n_jobs = -1)

lgb_model.fit(X_train, y_train, 

        eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=200)



print("훈련 점수: {:.2f}".format(lgb_model.score(X_train, y_train)))

print("테스트 점수: {:.2f}".format(lgb_model.score(X_test, y_test)))



eli5.show_weights(lgb_model, feature_filter=lambda x: x != '<BIAS>')
# 결과 제출

y_pred = lgb_model.predict(test[used_features])

submission['revenue'] = y_pred

submission.to_csv('TMDB_base_model_submission.csv', index=False)