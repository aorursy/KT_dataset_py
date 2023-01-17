import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import warnings

warnings.filterwarnings(action="ignore")

import os

#import featuretools as ft
print(os.listdir("../input"))
raw_data1 = pd.read_excel("../input/Data_Train.xlsx")

test_data = pd.read_excel("../input/Data_Test.xlsx")

raw_data1['source'] = 'train'

test_data['source'] = 'test'

raw_data = pd.concat([raw_data1,test_data],ignore_index=True)
raw_data.info(memory_usage='deep')
raw_data.columns
raw_data.head(5)

raw_data = raw_data[['CITY', 'CUISINES', 'LOCALITY', 'RATING',

       'TIME', 'TITLE', 'VOTES', 'source', 'COST']]
print("The train shape is", raw_data1.shape)

print("The test shape is", test_data.shape)
raw_data.isnull().sum()
#raw_data.drop(['RESTAURANT_ID'],inplace=True,axis=1)
raw_data.isnull().sum()
raw_data.head(5)
raw_data.dtypes
def unique_counts(df,features,p=False):

    for i in features:

        if p:

         print("The number of unique values for",i, df[i].value_counts())

         print("The number of unique values for",i, len(df[i].value_counts()))

         print("-"*100)

        else :

            print("The number of unique values for",i, len(df[i].value_counts()))

            print("-"*100)
ft = ['CITY','CUISINES','LOCALITY','TITLE',]

unique_counts(raw_data,ft)
raw_data['CITY'].fillna('NA', inplace=True)

raw_data['LOCALITY'].fillna('XXX', inplace=True)

raw_data['RATING'].fillna(0, inplace=True)

raw_data['VOTES'].fillna("0 votes",inplace=True)
def comma_seprated_categories(df, col):

    cols = list(df[col])

    max = 1

    for i in cols:

        if len(i.split(',')) > max:

            max = len(i.split(','))

    print("Max number of",col,"in a cell",max)

    

    all_cats = []

    for i in cols :

     if len(i.split(',')) == 1:

         all_cats.append(i.split(',')[0].strip().upper())

     else :

        for it in range(len(i.split(','))):

            all_cats.append(i.split(',')[it].strip().upper())

    print("\n\nNumber of Unique",col,": ", len(pd.Series(all_cats).unique()))

    print("\n\nUnique",col,":\n", pd.Series(all_cats).unique()) 

    return list(pd.Series(all_cats).unique())
all_titles = comma_seprated_categories(raw_data,col='TITLE')

all_cuisines =comma_seprated_categories(raw_data,col='CUISINES')
raw_data.isnull().sum()
#comma_seprated_categories(raw_data,col='TITLE')

comma_seprated_categories(raw_data,col='CUISINES')
raw_data[raw_data['source']=='train']['COST'].describe()
sns.distplot((raw_data[raw_data['source']=='train']['COST']))

plt.show()
np.log(raw_data[raw_data['source']=='train']['COST']).describe()
sns.distplot(np.log(raw_data[raw_data['source']=='train']['COST']))

plt.show()
print(len(raw_data['CITY'].str.split(" ",expand = True)[0].value_counts()))

print(len(raw_data['CITY'].value_counts()))
raw_data['CUISINES'].str.split(",",expand = True).head(5)

# Max 8 cusines in any given row
len(raw_data['TITLE'].unique())
raw_data['TITLE_1'] = raw_data['TITLE'].str.split(",",expand = True)[0]

raw_data['TITLE_2'] = raw_data['TITLE'].str.split(",",expand = True)[1]
raw_data.drop(['TITLE'],inplace=True,axis=1)
raw_data.head(5)
len( raw_data['LOCALITY'].str.split(" ",expand = True)[0].unique())
def mapping(df,col,n=25):

 print(col,n)

 vc = df[col].value_counts()

 replacements = {}

 for col, s in vc.items():

    if s[s<n].any():

        replacements[col] = 'other'

 return replacements
sns.boxplot(x='TITLE_1',y='COST', data=raw_data)

sns.set(rc={'figure.figsize':(40,30)})

sns.set(font_scale=1)   

plt.show()
raw_data.isnull().sum()
raw_data['TITLE_2'].fillna("None", inplace=True)
raw_data.isnull().sum()
raw_data['CUISINES_1'] = raw_data['CUISINES'].str.split(",",expand = True)[0]

raw_data['CUISINES_2'] = raw_data['CUISINES'].str.split(",",expand = True)[1]

raw_data['CUISINES_3'] = raw_data['CUISINES'].str.split(",",expand = True)[2]

raw_data['CUISINES_4'] = raw_data['CUISINES'].str.split(",",expand = True)[3]

raw_data['CUISINES_5'] = raw_data['CUISINES'].str.split(",",expand = True)[4]

raw_data['CUISINES_6'] = raw_data['CUISINES'].str.split(",",expand = True)[5]

raw_data['CUISINES_7'] = raw_data['CUISINES'].str.split(",",expand = True)[6]

raw_data['CUISINES_8'] = raw_data['CUISINES'].str.split(",",expand = True)[7]
raw_data.isnull().sum()
cus_list = []

for i in range(1,9):

    i = str(i)

    cus_list.append("CUISINES_"+i)

    
for i in cus_list:

    raw_data[i].fillna("NAA", inplace=True)

all_cuisines.append("NAA")
raw_data.isnull().sum()
raw_data.dtypes
raw_data['VOTES'] = raw_data['VOTES'].str.split(" ", expand=True)[0]

raw_data['VOTES']  = pd.to_numeric(raw_data['VOTES'])
raw_data.dtypes
rates = list(raw_data['RATING'])



for i in range(len(rates)) :

    try:

       rates[i] = float(rates[i])

    except :

       rates[i] = np.nan
raw_data['RATING'] = rates
raw_data.isnull().sum()
raw_data['RATING'].fillna(0.0,inplace=True)
raw_data.isnull().sum()
raw_data.dtypes
raw_data.drop(['TIME'], inplace=True,axis=1)
raw_data.drop(['CUISINES'], inplace=True,axis=1)
raw_data.columns
raw_data = raw_data[['CITY', 'LOCALITY', 'RATING', 'VOTES', 'source', 'TITLE_1',

       'TITLE_2', 'CUISINES_1', 'CUISINES_2', 'CUISINES_3', 'CUISINES_4',

       'CUISINES_5', 'CUISINES_6', 'CUISINES_7', 'CUISINES_8','COST']]
raw_data.head(10)
#correlation matrix

corrmat = raw_data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);

plt.show()
raw_data.corr()
#scatter plot experince/saleprice



var = 'RATING'

data = pd.concat([raw_data['COST'], raw_data[var]], axis=1)

#plt.figure(figsize=(40,20))

#plt.xlabel('xlabel', fontsize=50)

#plt.ylabel('ylabel', fontsize=50)

data.plot.scatter(x=var, y='COST', figsize = (40,20), s=120,grid=True,fontsize=40,);

plt.show()
plt.figure(figsize=(40,20))

plt.xlabel('xlabel', fontsize=50)

plt.ylabel('ylabel', fontsize=50)

sns.distplot(raw_data["RATING"])

plt.show()
raw_data['RATING'].describe()
raw_data['VOTES'].describe()
plt.figure(figsize=(40,20))

plt.xlabel('xlabel', fontsize=50)

plt.ylabel('ylabel', fontsize=50)

sns.distplot(raw_data["VOTES"])

plt.show()
raw_data.isnull().sum()
def mapping(df,col,n=25):

 print(col,n)

 vc = df[col].value_counts()

 replacements = {}

 for col, s in vc.items():

    if s[s<n].any():

        replacements[col] = 'other'

 return replacements

local = mapping(raw_data,'LOCALITY',n=40)

raw_data['LOCALITY'] = raw_data['LOCALITY'].replace(local)
from sklearn.preprocessing import LabelEncoder





le_titles = LabelEncoder()

le_cuisines = LabelEncoder()

le_city = LabelEncoder()

le_locality = LabelEncoder()





le_titles.fit(all_titles)

le_cuisines.fit(all_cuisines)



le_city.fit(raw_data['CITY'])

le_locality.fit(raw_data['LOCALITY'])
raw_data['TITLE_1']=raw_data['TITLE_1'].str.upper()

raw_data['TITLE_2']=raw_data['TITLE_2'].str.upper()
for i in cus_list:

    raw_data[i] = raw_data[i].str.upper()

    raw_data[i] = raw_data[i].str.strip()
raw_data['TITLE_1'] = le_titles.transform(raw_data['TITLE_1'])

raw_data['TITLE_2'] = le_titles.transform(raw_data['TITLE_2'])





raw_data['CUISINES_1'] = le_cuisines.transform(raw_data['CUISINES_1'])

raw_data['CUISINES_2'] = le_cuisines.transform(raw_data['CUISINES_2'])

raw_data['CUISINES_3'] = le_cuisines.transform(raw_data['CUISINES_3'])

raw_data['CUISINES_4'] = le_cuisines.transform(raw_data['CUISINES_4'])

raw_data['CUISINES_5'] = le_cuisines.transform(raw_data['CUISINES_5'])

raw_data['CUISINES_6'] = le_cuisines.transform(raw_data['CUISINES_6'])

raw_data['CUISINES_7'] = le_cuisines.transform(raw_data['CUISINES_7'])

raw_data['CUISINES_8'] = le_cuisines.transform(raw_data['CUISINES_8'])





raw_data['CITY'] = le_city.transform(raw_data['CITY'])

raw_data['LOCALITY'] = le_locality.transform(raw_data['LOCALITY'])
raw_data.head(5)
le_cuisines.inverse_transform([88])
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

raw_data['VOTES']= sc.fit_transform(raw_data['VOTES'].reshape(len(raw_data['VOTES']),1))

raw_data['RATING']= sc.fit_transform(raw_data['RATING'].reshape(len(raw_data['RATING']),1))
raw_data.head(5)
raw_data.drop(["CUISINES_6","CUISINES_7","CUISINES_8"], axis=1,inplace=True)
raw_data.head(5)
#Divide into test and train:

train = raw_data.loc[raw_data['source']=="train"]

test = raw_data.loc[raw_data['source']=="test"]

train.drop(["source"], inplace=True, axis=1)

test.drop(["source","COST"], inplace=True, axis=1)
train['COST'] = np.log(train['COST'])
from sklearn import model_selection

from sklearn.metrics import mean_squared_error

import xgboost

import numpy as np

from sklearn.ensemble import RandomForestRegressor

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train.drop(["COST"], axis=1), train['COST'])
from sklearn.grid_search import GridSearchCV
RF_G= param_grid={

            'max_depth': [4,8,10,12,14,16],

            'n_estimators': [10,20,30,40,50],

            'min_samples_split': [2, 5, 10]

        },
gsearchRF1 = GridSearchCV(estimator = RandomForestRegressor(),param_grid=RF_G,scoring='neg_mean_squared_log_error')

gsearchRF1.fit(train_x,train_y)
gsearchRF1.best_params_
RF_G_2 = param_grid={

            'max_depth': [12],

            'n_estimators': [40,50,60],

            'min_samples_split': [10,12,15,19,24]

        },
gsearchRF2 = GridSearchCV(estimator = RandomForestRegressor(),param_grid=RF_G_2,scoring='neg_mean_squared_log_error')

gsearchRF2.fit(train_x,train_y)
gsearchRF2.best_params_
RF1 = RandomForestRegressor(max_depth=14, min_samples_split=10, n_estimators=50, n_jobs=-1)
RF1.fit(train_x,train_y)
RF1.score(valid_x,valid_y)
sub1 = np.exp(RF1.predict(test))
pd.DataFrame(sub1).to_excel("./subimission_01.xlsx")
import xgboost
param_test1 = {'n_estimators':list(range(20,121,10))}

gsearch1 = GridSearchCV(estimator = xgboost.XGBRegressor(learning_rate=0.1, min_samples_split=100,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 

param_grid = param_test1, scoring='neg_mean_squared_log_error',n_jobs=4,iid=False, cv=5)

%time gsearch1.fit(train_x,train_y)
gsearch1.best_params_
param_test2 = {'max_depth':list(range(5,16,2))}

gsearch2 = GridSearchCV(estimator = xgboost.XGBRegressor(learning_rate=0.1, n_estimators=120, max_features='sqrt', subsample=0.8, random_state=10), 

param_grid = param_test2, scoring='neg_mean_squared_log_error',n_jobs=4,iid=False, cv=5)

gsearch2.fit(train_x,train_y)

#print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
gsearch2.best_params_
param_test3 = {'min_child_weight':[1,2,3,4]}

gsearch3 = GridSearchCV(estimator = xgboost.XGBRegressor(learning_rate=0.1, n_estimators=120, max_features='sqrt', subsample=0.8, random_state=10,max_depth=7), 

param_grid = param_test3, scoring='neg_mean_squared_log_error',n_jobs=4,iid=False, cv=5)

gsearch3.fit(train_x,train_y)

#print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
gsearch3.best_params_
param_test4 = {'gamma':[i/10.0 for i in range(1,10)]}

gsearch4 = GridSearchCV(estimator = xgboost.XGBRegressor(learning_rate=0.1, n_estimators=120, max_features='sqrt', subsample=0.8, random_state=10,max_depth=7, min_child_weight=3), 

param_grid = param_test4, scoring='neg_mean_squared_log_error',n_jobs=4,iid=False, cv=5)

gsearch4.fit(train_x,train_y)
gsearch4.best_params_
xg_algo = xgboost.XGBRegressor(n_estimators=120, max_depth=7, min_child_weight=3,

                              gamma=0.6)
xg_algo.fit(train_x,train_y)
xg_algo.score(valid_x,valid_y)
sub9 = np.exp(xg_algo.predict(test))

pd.DataFrame(sub9).to_excel("./subimission_09.xlsx")
xgboost.plot_importance(xg_algo)

plt.figure(figsize=(40,20))

#plt.xlabel(fontsize=50)

#plt.ylabel(fontsize=50)

plt.show()
sub2 = np.exp(xg_algo.predict(test))

pd.DataFrame(sub2).to_excel("./subimission_02.xlsx")
sub1
sub2
param_test5 = {'n_estimators':list(range(120,180,10)),

              'max_depth':list(range(5,16,2)),

              'min_child_weight':[1,2,3,4],

              'gamma':[i/10.0 for i in range(1,10)]}
gsearch5 = GridSearchCV(estimator = xgboost.XGBRegressor(learning_rate=0.1, max_features='sqrt', subsample=0.8, random_state=10), 

param_grid = param_test5, scoring='neg_mean_squared_log_error',n_jobs=-1,iid=False, cv=5)

gsearch5.fit(train_x,train_y)
gsearch5.best_params_
xg_boost2 = xgboost.XGBRegressor(max_depth=11, gamma=0.6,min_child_weight=2, n_estimators=120)
xg_boost2.fit(train_x, train_y)
xg_boost2.score(valid_x, valid_y)
sub9 = np.exp(xg_boost2.predict(test))

pd.DataFrame(sub9).to_excel("./subimission_09.xlsx")
xgboost.plot_importance(xg_boost2)

plt.show()
sub3 = (sub2 + sub4) /  2

pd.DataFrame(sub2).to_excel("./subimission_03.xlsx")
sub5 = np.exp(xg_boost2.predict(test))

#pd.DataFrame(sub5).to_excel("./subimission_05.xlsx")
sub5
from keras import layers

from keras import models

from keras.layers import Dropout

def build_model():

    model = models.Sequential()

    model.add(layers.Dense(64, activation = 'relu', input_shape = (train_x.shape[1],)))

    model.add(Dropout(.2))

    model.add(layers.Dense(64, activation = 'relu'))

    model.add(Dropout(.2))

    model.add(layers.Dense(64, activation = 'relu'))

    model.add(Dropout(.2))

    model.add(layers.Dense(64, activation = 'relu'))

    model.add(layers.Dense(1))

    model.add(layers.Dense(64, activation = 'relu'))

    model.add(layers.Dense(1))

    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mse'])

    return model
NN_model = build_model()
NN_model.fit(train_x, train_y, epochs=30, verbose=1,validation_data=(valid_x,valid_y))
sub6 = np.exp(NN_model.predict(test))
sub66 = (sub5 + sub6)/2

pd.DataFrame(sub66).to_excel("./sub66.xlsx")
import pandas as pd
pd.__version__