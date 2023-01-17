# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import lightgbm as lgb

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error,roc_auc_score

from google.cloud import bigquery

from sklearn.model_selection import KFold, StratifiedKFold

from datetime import date

from datetime import timedelta

import gc
def MyLabelEncode(coltr,colte):

    levels=coltr.unique().tolist()

    for l in levels:

        if l is np.nan:

            levels.remove(np.nan)

    levelmap={e:i for i,e in enumerate(levels)}

    return coltr.map(levelmap),colte.map(levelmap)
def MyLabelEncodeSingle(col):

    levels=col.unique().tolist()

    for l in levels:

        if l is np.nan:

            levels.remove(np.nan)

    levelmap={e:i for i,e in enumerate(levels)}

    return col.map(levelmap)
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
X_train = train.drop(["Fatalities", "ConfirmedCases"], axis=1)
countries = X_train["Country_Region"]
X_train = X_train.drop(["Id"], axis=1)

X_test = test.drop(["ForecastId"], axis=1)
X_train['Province_State']=X_train['Province_State'].fillna('Unknown')

X_test['Province_State']=X_test['Province_State'].fillna('Unknown')
X_train.dtypes
X_train['UpToNow']=(pd.to_datetime(date.today())-pd.to_datetime(X_train['Date'])).dt.days.astype(float)

X_test['UpToNow']=(pd.to_datetime(date.today())-pd.to_datetime(X_test['Date'])).dt.days.astype(float)
X_train['Date']= pd.to_datetime(X_train['Date']) 

X_test['Date']= pd.to_datetime(X_test['Date']) 

#X_train = X_train.set_index(['Date'])

#X_test = X_test.set_index(['Date'])
def create_time_features(df):

    """

    Creates time series features from datetime index

    """

    #df['date'] = df.index

    df['hour'] = df['Date'].dt.hour

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['quarter'] = df['Date'].dt.quarter

    df['month'] = df['Date'].dt.month

    df['year'] = df['Date'].dt.year

    df['dayofyear'] = df['Date'].dt.dayofyear

    df['dayofmonth'] = df['Date'].dt.day

    df['weekofyear'] = df['Date'].dt.weekofyear

    

    X = df[['hour','dayofweek','quarter','month','year',

           'dayofyear','dayofmonth','weekofyear']]

    return X
create_time_features(X_train)

create_time_features(X_test)
world_happiness_index = pd.read_csv("../input/world-bank-datasets/World_Happiness_Index.csv")
world_happiness_index=world_happiness_index.iloc[:,:19]
world_happiness_index.columns=[c.replace('(','').replace(')','').replace('(','').replace(',','').replace('-','_').replace('/','_').replace(' ','_') 

                               for c in world_happiness_index.columns]
average_year={}

temp_matrix=world_happiness_index.iloc[:,2:]

for y in world_happiness_index.Year.unique():

    average_year[y]=temp_matrix.loc[world_happiness_index.Year==y,:].mean()

del temp_matrix

gc.collect()
distance=0

while world_happiness_index.isna().sum().sum()!=0:

    for y in world_happiness_index.Year.unique():

        yhat=y-distance

        if yhat>2018:

            yhat=2018

        elif yhat<2005:

            yhat=2005

        for c in world_happiness_index.columns[2:]:

            world_happiness_index.loc[world_happiness_index.Year==y,c]=world_happiness_index.loc[world_happiness_index.Year==y,c].fillna(average_year[yhat][c])

        yhat=y+distance

        if yhat>2018:

            yhat=2018

        elif yhat<2005:

            yhat=2005

        for c in world_happiness_index.columns[2:]:

            world_happiness_index.loc[world_happiness_index.Year==y,c]=world_happiness_index.loc[world_happiness_index.Year==y,c].fillna(average_year[yhat][c])

        distance += 1
world_happiness_latest = world_happiness_index.groupby('Country_name').nth(-1)
world_happiness_first = world_happiness_index.groupby('Country_name').agg('first')
world_happiness_last = world_happiness_index.groupby('Country_name').agg('last')
world_happiness_count = world_happiness_index.groupby('Country_name').count()
world_happiness_range=(world_happiness_last-world_happiness_first)/world_happiness_count
world_happiness_range.drop("Year", axis=1, inplace=True)

world_happiness_latest.drop("Year", axis=1, inplace=True)

world_happiness_range.columns=[c+'_range' for c in world_happiness_range.columns]

world_happiness_latest.columns=[c+'_latest' for c in world_happiness_latest.columns]

world_happiness_grouped=pd.concat((world_happiness_latest,world_happiness_range),axis=1).reset_index()
X_train = pd.merge(left=X_train, right=world_happiness_grouped, how='left', left_on='Country_Region', right_on='Country_name')

X_test = pd.merge(left=X_test, right=world_happiness_grouped, how='left', left_on='Country_Region', right_on='Country_name')
X_train.drop("Country_name", axis=1, inplace=True)

X_test.drop("Country_name", axis=1, inplace=True)
malaria_world_health = pd.read_csv("../input/world-bank-datasets/Malaria_World_Health_Organization.csv")
malaria_world_health.columns=[c.replace(' ','_') for c in malaria_world_health.columns]
X_train = pd.merge(left=X_train, right=malaria_world_health, how='left', left_on='Country_Region', right_on='Country')

X_test = pd.merge(left=X_test, right=malaria_world_health, how='left', left_on='Country_Region', right_on='Country')
X_train.drop("Country", axis=1, inplace=True)

X_test.drop("Country", axis=1, inplace=True)
human_development_index = pd.read_csv("../input/world-bank-datasets/Human_Development_Index.csv")
human_development_index.columns=[c.replace(')','').replace('(','').replace(' ','_') for c in human_development_index.columns]
X_train = pd.merge(left=X_train, right=human_development_index, how='left', left_on='Country_Region', right_on='Country')

X_test = pd.merge(left=X_test, right=human_development_index, how='left', left_on='Country_Region', right_on='Country')
X_train.drop("Country", axis=1, inplace=True)

X_test.drop("Country", axis=1, inplace=True)
night_ranger_predictors = pd.read_csv("../input/covid19-demographic-predictors/covid19_by_country.csv")
night_ranger_predictors.columns=[c.replace(' ','_') for c in night_ranger_predictors.columns]
#There is a duplicate for Georgia in this dataset from Night Ranger, causing merge issues so we will just drop the Georgia rows

night_ranger_predictors = night_ranger_predictors[night_ranger_predictors.Country != "Georgia"]
night_ranger_predictors=night_ranger_predictors[['Country','Median_Age','GDP_2018','Crime_Index','Population_2020','Smoking_2016','Females_2018']]
X_train = pd.merge(left=X_train, right=night_ranger_predictors, how='left', left_on='Country_Region', right_on='Country')

X_test = pd.merge(left=X_test, right=night_ranger_predictors, how='left', left_on='Country_Region', right_on='Country')

X_train.drop("Country", axis=1, inplace=True)

X_test.drop("Country", axis=1, inplace=True)
X_train['Gross_national_income_GNI_per_capita_2018']= X_train['Gross_national_income_GNI_per_capita_2018'].apply(lambda x: x if x!=x else x.replace(',','')).astype(float)

X_test['Gross_national_income_GNI_per_capita_2018']= X_test['Gross_national_income_GNI_per_capita_2018'].apply(lambda x: x if x!=x else x.replace(',','')).astype(float)
weather_df = pd.read_csv("../input/covid19formattedweatherjan22march24/covid_dataset.csv")
weather_df=weather_df[['Province/State',

'Country/Region',

'lat',

'long',

'day',

'pop',

'urbanpop',

'density',

'medianage',

'smokers',

'health_exp_pc',

'hospibed',

'temperature',

'humidity']]
weather_df['Province/State']=weather_df['Province/State'].fillna('Unknown')
weather_df['day']=pd.to_datetime('2020-01-22')+weather_df['day'].apply(lambda x: timedelta(days=x))
weather_df['month']=weather_df['day'].dt.month

weather_df.drop('day',axis=1,inplace=True)
weather_df=weather_df.groupby(['Province/State','Country/Region','month']).mean().reset_index()
weather_df=weather_df.replace(-999,np.nan)
weather_df_latest = weather_df.groupby(['Province/State','Country/Region']).nth(-1).reset_index()

weather_df_latest['month']=4
weather_df=pd.concat((weather_df,weather_df_latest),sort=True,axis=0,ignore_index=True)
X_train = pd.merge(left=X_train, right=weather_df, how='left', left_on=['Country_Region','Province_State','month'], right_on=['Country/Region','Province/State','month'])

X_test = pd.merge(left=X_test, right=weather_df, how='left', left_on=['Country_Region','Province_State','month'], right_on=['Country/Region','Province/State','month'])

X_train.drop(['Country/Region','Province/State'], axis=1, inplace=True)

X_test.drop(['Country/Region','Province/State'], axis=1, inplace=True)
happiness_df = pd.read_csv("../input/world-happiness-report-2020/WHR20_DataForFigure2.1.csv")
happiness_df.columns=[c.replace(':','').replace('+','').replace(' ','_') for c in happiness_df.columns]
happiness_df.columns
happiness_df['Regional_indicator']=MyLabelEncodeSingle(happiness_df['Regional_indicator'])
X_train = pd.merge(left=X_train, right=happiness_df, how='left', left_on='Country_Region', right_on='Country_name')

X_test = pd.merge(left=X_test, right=happiness_df, how='left', left_on='Country_Region', right_on='Country_name')

X_train.drop('Country_name', axis=1, inplace=True)

X_test.drop('Country_name', axis=1, inplace=True)
age_df = pd.read_csv("../input/world-population-by-age-group-2020/WorldPopulationByAge2020.csv")
age_df['AgeGrp']=MyLabelEncodeSingle(age_df['AgeGrp'])
def processAge(df):

    ageindex=df['AgeGrp']

    sexsum=df[['PopMale', 'PopFemale', 'PopTotal']].sum()

    mp=sexsum['PopMale']/sexsum['PopTotal']

    fp=sexsum['PopFemale']/sexsum['PopTotal']

    p0=df.loc[ageindex==0,'PopTotal'].values[0]/sexsum['PopTotal']

    p1=df.loc[ageindex==1,'PopTotal'].values[0]/sexsum['PopTotal']

    p2=df.loc[ageindex==2,'PopTotal'].values[0]/sexsum['PopTotal']

    p3=df.loc[ageindex==3,'PopTotal'].values[0]/sexsum['PopTotal']

    m0=df.loc[ageindex==0,'PopMale'].values[0]/sexsum['PopMale']

    m1=df.loc[ageindex==1,'PopMale'].values[0]/sexsum['PopMale']

    m2=df.loc[ageindex==2,'PopMale'].values[0]/sexsum['PopMale']

    m3=df.loc[ageindex==3,'PopMale'].values[0]/sexsum['PopMale']

    f0=df.loc[ageindex==0,'PopFemale'].values[0]/sexsum['PopFemale']

    f1=df.loc[ageindex==1,'PopFemale'].values[0]/sexsum['PopFemale']

    f2=df.loc[ageindex==2,'PopFemale'].values[0]/sexsum['PopFemale']

    f3=df.loc[ageindex==3,'PopFemale'].values[0]/sexsum['PopFemale']

    return pd.DataFrame({'MaleP':mp,'MaleP_0':m0,'MaleP_1':m1,'MaleP_2':m2,'MaleP_3':m3,'FemaleP':fp,

                         'FemaleP_0':f0,'FemaleP_1':f1,'FemaleP_2':f2,'FemaleP_3':f3,'PopTotal':sexsum['PopTotal'],

                         'Pop_0':p0,'Pop_1':p1,'Pop_2':p2,'Pop_3':p3},index=[0])
age_df=age_df.groupby('Location').apply(processAge).reset_index().drop('level_1',axis=1)
X_train = pd.merge(left=X_train, right=age_df, how='left', left_on='Country_Region', right_on='Location')

X_test = pd.merge(left=X_test, right=age_df, how='left', left_on='Country_Region', right_on='Location')

X_train.drop('Location', axis=1, inplace=True)

X_test.drop('Location', axis=1, inplace=True)
healthsys_df = pd.read_csv("../input/world-bank-wdi-212-health-systems/2.12_Health_systems.csv")

healthsys_df.columns=[c.replace('-','_') for c in healthsys_df.columns]
healthsys_df.drop('World_Bank_Name',axis=1,inplace=True)
nan_country=healthsys_df[['Country_Region', 'Province_State']].isna().all(axis=1)
healthsys_df=healthsys_df.loc[nan_country==False,:].reset_index(drop=True)
healthsys_df['Province_State']=healthsys_df['Province_State'].fillna('Unknown')
X_train = pd.merge(left=X_train, right=healthsys_df, how='left', left_on=['Country_Region','Province_State'], right_on=['Country_Region', 'Province_State'])

X_test = pd.merge(left=X_test, right=healthsys_df, how='left', left_on=['Country_Region','Province_State'], right_on=['Country_Region', 'Province_State'])
pop_df = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")

pop_df.columns=[c.replace('.',' ').split(' ')[0]+'_pop2020' for c in pop_df.columns]
percent_col=['Yearly_pop2020','Urban_pop2020', 'World_pop2020']

def depercent(x):

    if x=='N.A.':

        return np.nan 

    else:

        return float(x.replace('%',''))

for c in percent_col:

    pop_df[c]=pop_df[c].apply(lambda x: depercent(x))
pop_df=pop_df.replace('N.A.',np.nan)
pop_df[['Population_pop2020', 'Yearly_pop2020',

       'Net_pop2020', 'Density_pop2020', 'Land_pop2020', 'Migrants_pop2020',

       'Fert_pop2020', 'Med_pop2020', 'Urban_pop2020', 'World_pop2020']]=pop_df[['Population_pop2020', 'Yearly_pop2020',

       'Net_pop2020', 'Density_pop2020', 'Land_pop2020', 'Migrants_pop2020',

       'Fert_pop2020', 'Med_pop2020', 'Urban_pop2020', 'World_pop2020']].astype(float)
X_train = pd.merge(left=X_train, right=pop_df, how='left', left_on='Country_Region', right_on='Country_pop2020')

X_test = pd.merge(left=X_test, right=pop_df, how='left', left_on='Country_Region', right_on='Country_pop2020')

X_train.drop('Country_pop2020', axis=1, inplace=True)

X_test.drop('Country_pop2020', axis=1, inplace=True)
compre_df = pd.read_csv("../input/countryinfo/covid19countryinfo.csv")

#testcase_df = pd.read_csv("../input/countryinfo/covid19tests.csv")
compre_df['region']=compre_df['region'].fillna('Unknown')
keepcol=['region', 'country', 'tests',

       'testpop', 'density', 'medianage', 'urbanpop', 'quarantine', 'schools',

       'publicplace', 'gatheringlimit', 'gathering', 'nonessential',

       'hospibed', 'smokers', 'sex0', 'sex14', 'sex25', 'sex54', 'sex64',

       'sex65plus', 'sexratio', 'lung', 'femalelung', 'malelung', 'gdp2019',

       'healthexp', 'healthperpop', 'fertility', 'firstcase']
def tempfun(x):

    if x is np.nan:

        return x

    else:

        return float(x.replace(',',''))

for c in ['gdp2019','healthexp']:

    compre_df[c]=compre_df[c].apply(lambda x: tempfun(x) )
todate_col=['quarantine', 'schools','publicplace', 'gathering', 'nonessential','firstcase']

for c in todate_col:

    compre_df[c]= (pd.to_datetime(date.today())-pd.to_datetime(compre_df[c])).dt.days.astype(float)

compre_df=compre_df[keepcol]
X_train = pd.merge(left=X_train, right=compre_df, how='left', left_on=['Country_Region','Province_State'], right_on=['country','region'])

X_test = pd.merge(left=X_test, right=compre_df, how='left', left_on=['Country_Region','Province_State'], right_on=['country','region'])

X_train.drop(['country','region'], axis=1, inplace=True)

X_test.drop(['country','region'], axis=1, inplace=True)
X_train['Country_Region'],X_test['Country_Region']=MyLabelEncode(X_train['Country_Region'],X_test['Country_Region'])

X_train['Province_State'],X_test['Province_State']=MyLabelEncode(X_train['Province_State'],X_test['Province_State'])
f_cat=['Country_Region','Province_State','Regional_indicator']
target_F = train["Fatalities"].reset_index(drop=True)

target_C = train["ConfirmedCases"].reset_index(drop=True)

#X_train = X_train.reset_index(drop=True)

#X_test = X_test.reset_index(drop=True)
X_train.drop(['Date','year'], axis=1, inplace=True)

X_test.drop(['Date','year'], axis=1, inplace=True)
train_public_index=pd.DataFrame({'month':X_train['month']<=3,'day':X_train['dayofmonth']<19})

train_public_index=train_public_index.all(axis=1)

X_train_public=X_train.loc[train_public_index,:].reset_index(drop=True)
target_C_public=target_C[train_public_index].reset_index(drop=True)

target_F_public=target_F[train_public_index].reset_index(drop=True)
test_public_index=pd.DataFrame({'month':X_test['month']<=4,'day':X_test['dayofmonth']<=1})

test_public_index=test_public_index.all(axis=1)

test_private_index=~test_public_index

X_test_public=X_test.loc[test_public_index,:].reset_index(drop=True)

X_test_private=X_test.loc[test_private_index,:].reset_index(drop=True)
X_test.shape
X_test_private.shape
usedfeatures=X_train.columns.tolist()
def RMSLE(t1,p1):

    return np.sqrt(np.mean((np.log(t1+1)-np.log(p1+1))**2))
country_col=X_train[f_cat[0]]
import matplotlib.pyplot as plt

plt.hist(target_F_public)
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1990)
####################C+public

params = {#1

        'learning_rate': 0.5,

        'feature_fraction': 1,

        'min_data_in_leaf' : 150,

        'max_depth': 6,

#        'max_bin':300,

        'reg_alpha': 10,#l1

#        'reg_lambda': 10,#l2

        'num_leaves':15,

        'objective': 'regression',

        'metric': 'rmse',

        'n_jobs': -1,

        'feature_fraction_seed': 42,

        'bagging_seed': 42,

        'boosting_type': 'gbdt',

        'verbose': 1,

        'is_unbalance': False,

#        'bagging_freq':5,

#        'pos_bagging_fraction':0.8,

#        'neg_bagging_fraction':0.8,

        'boost_from_average': False}

traintion_public_c = np.zeros(len(X_train_public))

validation_public_c = np.zeros(len(X_train_public))

predictions_public_c = np.zeros(len(X_test_public))

feature_importance_df_public_c = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_public,target_C_public)):

    print("fold n°{}".format(fold_))

    train_x=X_train_public.iloc[trn_idx][usedfeatures].reset_index(drop=True)

    valid_x=X_train_public.iloc[val_idx][usedfeatures].reset_index(drop=True)

    target_train=target_C_public.iloc[trn_idx].reset_index(drop=True)

    target_valid=target_C_public.iloc[val_idx].reset_index(drop=True)

    trn_data = lgb.Dataset(train_x,

                           label=target_train,

                           categorical_feature=f_cat

                          )

    val_data = lgb.Dataset(valid_x,

                           label=target_valid,

                           categorical_feature=f_cat

                          )



    num_round = 1000000

    clf = lgb.train(params,

                    trn_data,

                    num_round,

                    valid_sets = [trn_data, val_data],

                    verbose_eval=250,

                    early_stopping_rounds = 150)

    traintion_public_c[trn_idx] += clf.predict(train_x, num_iteration=clf.best_iteration)/(folds.n_splits-1)

    validation_public_c[val_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = usedfeatures

    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df_public_c = pd.concat([feature_importance_df_public_c, fold_importance_df], axis=0)

    

    predictions_public_c += clf.predict(X_test_public, num_iteration=clf.best_iteration) / folds.n_splits

print("==========C+Pub==============")

print("Train RMSLE score: {:<8.5f}".format(RMSLE(target_C_public,traintion_public_c)))

print("Valid RMSLE score: {:<8.5f}".format(RMSLE(target_C_public,validation_public_c)))

####################C+private

params = {#1

        'learning_rate': 0.1,

        'feature_fraction': 1,

        'min_data_in_leaf' : 1,

        'max_depth': 8,

       'max_bin':350,

#        'reg_alpha': 0.05,#l1

#        'reg_lambda': 0.05,#l2

        'num_leaves':50,

        'objective': 'regression',

        'metric': 'rmse',

        'n_jobs': -1,

        'feature_fraction_seed': 42,

        'bagging_seed': 42,

        'boosting_type': 'gbdt',

        'verbose': 1,

        'is_unbalance': False,

#        'bagging_freq':5,

#        'pos_bagging_fraction':0.8,

#        'neg_bagging_fraction':0.8,

        'boost_from_average': False}

traintion_private_c = np.zeros(len(X_train))

validation_private_c = np.zeros(len(X_train))

predictions_private_c = np.zeros(len(X_test_private))

feature_importance_df_private_c = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train,target_C)):

    print("fold n°{}".format(fold_))

    train_x=X_train.iloc[trn_idx][usedfeatures].reset_index(drop=True)

    valid_x=X_train.iloc[val_idx][usedfeatures].reset_index(drop=True)

    target_train=target_C.iloc[trn_idx].reset_index(drop=True)

    target_valid=target_C.iloc[val_idx].reset_index(drop=True)

    trn_data = lgb.Dataset(train_x,

                           label=target_train,

                           categorical_feature=f_cat

                          )

    val_data = lgb.Dataset(valid_x,

                           label=target_valid,

                           categorical_feature=f_cat

                          )



    num_round = 1000000

    clf = lgb.train(params,

                    trn_data,

                    num_round,

                    valid_sets = [trn_data, val_data],

                    verbose_eval=250,

                    early_stopping_rounds = 200)

    traintion_private_c[trn_idx] += clf.predict(train_x, num_iteration=clf.best_iteration)/(folds.n_splits-1)

    validation_private_c[val_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = usedfeatures

    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df_private_c = pd.concat([feature_importance_df_private_c, fold_importance_df], axis=0)

    

    predictions_private_c += clf.predict(X_test_private, num_iteration=clf.best_iteration) / folds.n_splits

print("==========C+Pri==============")

print("Train RMSLE score: {:<8.5f}".format(RMSLE(target_C,traintion_private_c)))

print("Valid RMSLE score: {:<8.5f}".format(RMSLE(target_C,validation_private_c)))

####################F+public

params = {#1

        'learning_rate': 0.1,

        'feature_fraction': 0.8,

        'min_data_in_leaf' : 100,

        'max_depth': 7,

#        'max_bin':200,

        'reg_alpha': 5,#l1

#        'reg_lambda': 0.5,#l2

        'objective': 'regression',

        'num_leaves':30,

        'metric': 'rmse',

        'n_jobs': -1,

        'feature_fraction_seed': 42,

        'bagging_seed': 42,

        'boosting_type': 'gbdt',

        'verbose': 1,

        'is_unbalance': False,

#        'bagging_freq':5,

#        'pos_bagging_fraction':0.8,

#        'neg_bagging_fraction':0.8,

        'boost_from_average': False}

traintion_public_f = np.zeros(len(X_train_public))

validation_public_f = np.zeros(len(X_train_public))

predictions_public_f = np.zeros(len(X_test_public))

feature_importance_df_public_f = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_public,target_F_public)):

    print("fold n°{}".format(fold_))

    train_x=X_train_public.iloc[trn_idx][usedfeatures].reset_index(drop=True)

    valid_x=X_train_public.iloc[val_idx][usedfeatures].reset_index(drop=True)

    target_train=target_F_public.iloc[trn_idx].reset_index(drop=True)

    target_valid=target_F_public.iloc[val_idx].reset_index(drop=True)

    trn_data = lgb.Dataset(train_x,

                           label=target_train,

                           categorical_feature=f_cat

                          )

    val_data = lgb.Dataset(valid_x,

                           label=target_valid,

                           categorical_feature=f_cat

                          )



    num_round = 1000000

    clf = lgb.train(params,

                    trn_data,

                    num_round,

                    valid_sets = [trn_data, val_data],

                    verbose_eval=250,

                    early_stopping_rounds = 20)

    traintion_public_f[trn_idx] += clf.predict(train_x, num_iteration=clf.best_iteration)/(folds.n_splits-1)

    validation_public_f[val_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = usedfeatures

    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df_public_f = pd.concat([feature_importance_df_public_f, fold_importance_df], axis=0)

    

    predictions_public_f += clf.predict(X_test_public, num_iteration=clf.best_iteration) / folds.n_splits

print("==========F+Pub==============")

print("Train RMSLE score: {:<8.5f}".format(RMSLE(target_F_public,traintion_public_f)))

print("Valid RMSLE score: {:<8.5f}".format(RMSLE(target_F_public,validation_public_f)))

####################F+private

params = {#1

        'learning_rate': 0.1,

        'feature_fraction': 0.7,

        'min_data_in_leaf' : 1,

        'max_depth': 8,

        'max_bin':300,

#        'reg_alpha': 0.01,#l1

#        'reg_lambda':0.01,#l2

        'objective': 'regression',

        'num_leaves':30,

        'metric': 'rmse',

        'n_jobs': -1,

        'feature_fraction_seed': 42,

        'bagging_seed': 42,

        'boosting_type': 'gbdt',

        'verbose': 1,

        'is_unbalance': False,

#        'bagging_freq':5,

#        'pos_bagging_fraction':0.8,

#        'neg_bagging_fraction':0.8,

        'boost_from_average': False}

traintion_private_f = np.zeros(len(X_train))

validation_private_f = np.zeros(len(X_train))

predictions_private_f = np.zeros(len(X_test_private))

feature_importance_df_private_f = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train,target_F)):

    print("fold n°{}".format(fold_))

    train_x=X_train.iloc[trn_idx][usedfeatures].reset_index(drop=True)

    valid_x=X_train.iloc[val_idx][usedfeatures].reset_index(drop=True)

    target_train=target_F.iloc[trn_idx].reset_index(drop=True)

    target_valid=target_F.iloc[val_idx].reset_index(drop=True)

    trn_data = lgb.Dataset(train_x,

                           label=target_train,

                           categorical_feature=f_cat

                          )

    val_data = lgb.Dataset(valid_x,

                           label=target_valid,

                           categorical_feature=f_cat

                          )



    num_round = 1000000

    clf = lgb.train(params,

                    trn_data,

                    num_round,

                    valid_sets = [trn_data, val_data],

                    verbose_eval=250,

                    early_stopping_rounds = 350)

    traintion_private_f[trn_idx] += clf.predict(train_x, num_iteration=clf.best_iteration)/(folds.n_splits-1)

    validation_private_f[val_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = usedfeatures

    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df_private_f = pd.concat([feature_importance_df_private_f, fold_importance_df], axis=0)

    

    predictions_private_f += clf.predict(X_test_private, num_iteration=clf.best_iteration) / folds.n_splits

print("==========F+Pri==============")

print("Train RMSLE score: {:<8.5f}".format(RMSLE(target_F,traintion_private_f)))

print("Valid RMSLE score: {:<8.5f}".format(RMSLE(target_F,validation_private_f)))
sub = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")

sub.loc[test_public_index,"Fatalities"] = predictions_public_f

sub.loc[test_public_index,"ConfirmedCases"] = predictions_public_c

sub.loc[test_private_index,"Fatalities"] = predictions_private_f

sub.loc[test_private_index,"ConfirmedCases"] = predictions_private_c

sub.to_csv('submission.csv',index=False)