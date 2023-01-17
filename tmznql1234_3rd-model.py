# package

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import json

from collections import Counter

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
train['has_collection'] = train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)

test['has_collection'] = test['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)
train['num_genres'] = train['genres'].apply(lambda x: len(x) if x != {} else 0)

test['num_genres'] = test['genres'].apply(lambda x: len(x) if x != {} else 0)
# genres에서 genre의 이름만 추출하기

train['all_genres'] = train['genres'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')



# 모든 장르 추출하기

genre_iter = (set(x) for x in train.all_genres)

genre = sorted(set.union(*genre_iter))



# 장르로 더미변수 만들기

for g in genre:

    train['genre_' + g] = train['all_genres'].apply(lambda x: 1 if g in x else 0)
train['num_production_companies'] = train['production_companies'].apply(lambda x: len(x) if x != {} else 0)

test['num_production_companies'] = test['production_companies'].apply(lambda x: len(x) if x != {} else 0)
train['num_production_countries'] = train['production_countries'].apply(lambda x: len(x) if x != {} else 0)

test['num_production_countries'] = test['production_countries'].apply(lambda x: len(x) if x != {} else 0)
# list_of_cast_names = list(train['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)



train['num_cast'] = train['cast'].apply(lambda x: len(x) if x != {} else 0)

test['num_cast'] = test['cast'].apply(lambda x: len(x) if x != {} else 0)
# list_of_crew_names = list(train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

train['num_crew'] = train['crew'].apply(lambda x: len(x) if x != {} else 0)

test['num_crew'] = test['crew'].apply(lambda x: len(x) if x != {} else 0)
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
train['all_genres'] = train['genres'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')

test['all_genres'] = test['genres'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')
def genre_budget(df):

    dummies = train.all_genres.apply(lambda x: pd.Series([1] * len(x), index=list(x))).fillna(0, downcast='infer')

    totaldf = pd.DataFrame(columns=list(dummies.columns))

    for i in range(len(train.index)):

        for factor in list(totaldf.columns):

            if dummies[factor][i] == 1:

                totaldf.loc[len(totaldf.index), factor] = train['budget'][i]

  

    meandf = pd.DataFrame(columns=list(dummies.columns), index=['average'])

    for factor in list(totaldf.columns):

        meandf[factor] = round(totaldf[factor].mean())

    

    genre_budget = pd.DataFrame(index=list(df.index), columns = ['genre_budget'])

    for i in range(len(df.index)):

        dum = []

        for factor in list(totaldf.columns):

            if factor in df['all_genres'][i]:

                dum.append(meandf[factor]['average'])

                genre_budget['genre_budget'][i] = sum(dum)/len(dum)



    df['genre_budget'] = genre_budget.fillna(0)
genre_budget(train)

genre_budget(test)
def genre_revenue(df):

    dummies = train.all_genres.apply(lambda x: pd.Series([1] * len(x), index=list(x))).fillna(0, downcast='infer')

    totaldf = pd.DataFrame(columns=list(dummies.columns))

    for i in range(len(train.index)):

        for factor in list(totaldf.columns):

            if dummies[factor][i] == 1:

                totaldf.loc[len(totaldf.index), factor] = train['revenue'][i]

  

    meandf = pd.DataFrame(columns=list(dummies.columns), index=['average'])

    for factor in list(totaldf.columns):

        meandf[factor] = round(totaldf[factor].mean())

    

    genre_revenue = pd.DataFrame(index=list(df.index), columns = ['genre_revenue'])

    for i in range(len(df.index)):

        dum = []

        for factor in list(totaldf.columns):

            if factor in df['all_genres'][i]:

                dum.append(meandf[factor]['average'])

                genre_revenue['genre_revenue'][i] = sum(dum)/len(dum)



    df['genre_revenue'] = genre_revenue.fillna(0)
genre_revenue(train)

genre_revenue(test)
train['all_companies'] = train['production_companies'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')

test['all_companies'] = test['production_companies'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')



list_of_company_names = list(train['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

top_company_names = [m[0] for m in Counter([i for j in list_of_company_names for i in j]).most_common(50)]



top_companies = []

for i in range(len(train.index)):

    names = set(train.all_companies[i])

    topnames = set(top_company_names)

    top_companies.append(names&topnames)

    

train['top_companies'] = top_companies
def company_budget(df):

    dummies = train.top_companies.apply(lambda x: pd.Series([1] * len(x), index=list(x))).fillna(0, downcast='infer')

    totaldf = pd.DataFrame(columns=list(dummies.columns))

    for i in range(len(train.index)):

        for factor in list(totaldf.columns):

            if dummies[factor][i] == 1:

                totaldf.loc[len(totaldf.index), factor] = train['budget'][i]

  

    meandf = pd.DataFrame(columns=list(dummies.columns), index=['average'])

    for factor in list(totaldf.columns):

        meandf[factor] = round(totaldf[factor].mean())

    

    company_budget = pd.DataFrame(index=list(df.index), columns = ['company_budget'])

    for i in range(len(df.index)):

        dum = []

        for factor in list(totaldf.columns):

            if factor in df['all_companies'][i]:

                dum.append(meandf[factor]['average'])

                company_budget['company_budget'][i] = sum(dum)/len(dum)



    df['company_budget'] = company_budget.fillna(0)
company_budget(train)

company_budget(test)
def company_revenue(df):

    dummies = train.top_companies.apply(lambda x: pd.Series([1] * len(x), index=list(x))).fillna(0, downcast='infer')

    totaldf = pd.DataFrame(columns=list(dummies.columns))

    for i in range(len(train.index)):

        for factor in list(totaldf.columns):

            if dummies[factor][i] == 1:

                totaldf.loc[len(totaldf.index), factor] = train['revenue'][i]

  

    meandf = pd.DataFrame(columns=list(dummies.columns), index=['average'])

    for factor in list(totaldf.columns):

        meandf[factor] = round(totaldf[factor].mean())

    

    company_revenue = pd.DataFrame(index=list(df.index), columns = ['company_revenue'])

    for i in range(len(df.index)):

        dum = []

        for factor in list(totaldf.columns):

            if factor in df['all_companies'][i]:

                dum.append(meandf[factor]['average'])

                company_revenue['company_revenue'][i] = sum(dum)/len(dum)



    df['company_revenue'] = company_revenue.fillna(0)
company_revenue(train)

company_revenue(test)
train['all_cast'] = train['cast'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')

test['all_cast'] = test['cast'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')



list_of_cast_names = list(train['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(50)]



top_casts = []

for i in range(len(train.index)):

    names = set(train.all_cast[i])

    topnames = set(top_cast_names)

    top_casts.append(names&topnames)

    

train['top_casts'] = top_casts
def cast_revenue(df):

    dummies = train.top_casts.apply(lambda x: pd.Series([1] * len(x), index=list(x))).fillna(0, downcast='infer')

    totaldf = pd.DataFrame(columns=list(dummies.columns))

    # 기존 train의 값

    for i in range(len(train.index)):

        for factor in list(totaldf.columns):

            if dummies[factor][i] == 1:

                totaldf.loc[len(totaldf.index), factor] = train['revenue'][i]

    meandf = pd.DataFrame(columns=list(dummies.columns), index=['average'])

    for factor in list(totaldf.columns):

        meandf[factor] = round(totaldf[factor].mean())

    # test에 적용하기

    cast_revenue = pd.DataFrame(index=list(df.index), columns = ['cast_revenue'])

    for i in range(len(df.index)):

        dum = []

        for factor in list(totaldf.columns):

            if factor in df['all_cast'][i]:

                dum.append(meandf[factor]['average'])

                cast_revenue['cast_revenue'][i] = sum(dum)/len(dum)



    df['cast_revenue'] = cast_revenue.fillna(0)
cast_revenue(train)

cast_revenue(test)
train['all_crew'] = train['crew'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')

test['all_crew'] = test['crew'].apply(lambda x: sorted([i['name'] for i in x]) if x != {} else '')



list_of_crew_names = list(train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(50)]



top_crews = []

for i in range(len(train.index)):

    names = set(train.all_crew[i])

    topnames = set(top_crew_names)

    top_crews.append(names&topnames)

train['top_crews'] = top_crews
def crew_revenue(df):

    dummies = train.top_crews.apply(lambda x: pd.Series([1] * len(x), index=list(x))).fillna(0, downcast='infer')

    totaldf = pd.DataFrame(columns=list(dummies.columns))



    for i in range(len(train.index)):

        for factor in list(totaldf.columns):

            if dummies[factor][i] == 1:

                totaldf.loc[len(totaldf.index), factor] = train['revenue'][i]



    meandf = pd.DataFrame(columns=list(dummies.columns), index=['average'])

    for factor in list(totaldf.columns):

        meandf[factor] = round(totaldf[factor].mean())

        

    # test에 적용하기

    crew_revenue = pd.DataFrame(index=list(train.index), columns = ['crew_revenue'])



    for i in range(len(df.index)):

        dum = []

        for factor in list(totaldf.columns):

            if factor in df['all_crew'][i]:

                dum.append(meandf[factor]['average'])

                crew_revenue['crew_revenue'][i] = sum(dum)/len(dum)



    df['crew_revenue'] = crew_revenue.fillna(0)
crew_revenue(train)

crew_revenue(test)
# 2/20/15로 되어있는 release date를 연도로 모으기

train[['release_month','release_day','release_year']]=train['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)

# 연도 끝 두자리수만 있기 때문에 앞에 19/20 붙이기

train.loc[ (train['release_year'] <= 19) & (train['release_year'] < 100), "release_year"] += 2000

train.loc[ (train['release_year'] > 19)  & (train['release_year'] < 100), "release_year"] += 1900



releaseDate = pd.to_datetime(train['release_date'])

train['release_dayofweek'] = releaseDate.dt.dayofweek

train['release_quarter'] = releaseDate.dt.quarter



# test

test[['release_month','release_day','release_year']]=test['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)

test.loc[ (train['release_year'] <= 19) & (test['release_year'] < 100), "release_year"] += 2000

test.loc[ (train['release_year'] > 19)  & (test['release_year'] < 100), "release_year"] += 1900



releaseDate = pd.to_datetime(test['release_date'])

test['release_dayofweek'] = releaseDate.dt.dayofweek

test['release_quarter'] = releaseDate.dt.quarter
release_month_rev = train.groupby("release_month")["revenue"].aggregate('mean')



for month in range(1, 13):

    train.loc[train['release_month'] == month, 'month_revenue'] = release_month_rev[month]

    

# test 적용

for month in range(1, 13):

    test.loc[test['release_month'] == month, 'month_revenue'] = release_month_rev[month]
release_dayofweek_rev = train.groupby("release_dayofweek")["revenue"].aggregate('mean')



for day in range(0,7):

    train.loc[train['release_dayofweek'] == day, 'day_revenue'] = release_dayofweek_rev[day]

    

# test 적용

for day in range(0,7):

    test.loc[test['release_dayofweek'] == day, 'day_revenue'] = release_dayofweek_rev[day]
or_lan_revenue = train.groupby('original_language')['revenue'].aggregate('mean')



for lan in or_lan_revenue.index:

    train.loc[train['original_language'] == lan, 'or_lan_revenue'] = or_lan_revenue["{}".format(lan)]

    

for lan in or_lan_revenue.index:

    test.loc[test['original_language'] == lan, 'or_lan_revenue'] = or_lan_revenue["{}".format(lan)]
train.num_cast[train.num_cast > 20] = 20

train.num_crew[train.num_crew > 20] = 20

train.num_Keywords[train.num_Keywords > 30] = 30



# test 적용

test.num_cast[test.num_cast > 20] = 20

test.num_crew[test.num_crew > 20] = 20

test.num_Keywords[test.num_Keywords > 30] = 30
# Formating for modeling
used_features = ['release_year', 'num_genres', 'num_production_companies', 'num_production_countries', 'num_cast',

                 'num_crew', 'num_spoken_languages', 'num_Keywords',  'has_homepage', 'isTaglineNA', 'isTitleDifferent',

                 'budget', 'runtime', 'isOriginalLanguageEng', 'isSpokenLanguageEng', 'has_collection',

                 'month_revenue', 'day_revenue', 'cast_revenue', 'crew_revenue','genre_budget', 'company_budget',

                 'genre_revenue', 'company_revenue', 'or_lan_revenue']



X = train[used_features]

y = train['revenue']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lgb_model = lgb.LGBMRegressor(n_estimators = 10000, nthread = 4, n_jobs = -1)

lgb_model.fit(X_train, y_train, 

        eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='rmse',

        verbose=1000, early_stopping_rounds=200)



print("{}".format(lgb_model))

print("훈련 점수: {:.2f}".format(lgb_model.score(X_train, y_train)))

print("테스트 점수: {:.2f}".format(lgb_model.score(X_test, y_test)))



eli5.show_weights(lgb_model, feature_filter=lambda x: x != '<BIAS>')
params = {'learning_rate': 0.1, 

          'max_depth': 5,

          'n_estimators': 10000, 

          'num_leaves': 31,

          'reg_alpha': 10000, 

          'reg_lambda': 0.0,

           }



# parameter 조정

lgb_model.set_params(**params)
print("{}".format(lgb_model))

print("훈련 점수: {:.2f}".format(lgb_model.score(X_train, y_train)))

print("테스트 점수: {:.2f}".format(lgb_model.score(X_test, y_test)))
# 결과 제출

y_pred = lgb_model.predict(test[used_features])

submission['revenue'] = y_pred

submission.to_csv('TMDB_base_model_submission03.csv', index=False)