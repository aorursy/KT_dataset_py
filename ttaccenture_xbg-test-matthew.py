import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns # visualization

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

import eli5

from eli5.sklearn import PermutationImportance

from eli5 import show_weights

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

import nltk

import string

from wordcloud import WordCloud, STOPWORDS

import re

from nltk.tokenize import RegexpTokenizer 

from collections import Counter

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from sklearn import datasets

from IPython.display import Image  

from sklearn import tree
df = pd.read_csv('../input/austin-bike/austin_bikeshare_trips.csv')
df.head()
df.dtypes
df['yearC'] = df['year'].astype('category')

df['monthC'] = df['month'].astype('category')

df['trip_idC'] = df['trip_id'].astype('category')



df['year_code'] = df['yearC'].cat.codes

df['month_code'] = df['monthC'].cat.codes

df['trip_code'] = df['trip_idC'].cat.codes
df.dtypes
x = 0

Date=[]

while x<len(df):

    Date.append(df.start_time[x][8:10])

    x = x+1
df['Date'] = Date

df
df['DateC'] = df['Date'].astype('category')



df['date_code'] = df['DateC'].cat.codes
y = df.date_code

features = ['year_code','month_code','trip_code']

x = df[features]

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)

basic_model = DecisionTreeRegressor()

basic_model.fit(train_x, train_y)

val_predictions = basic_model.predict(val_x)

print("Printing MAE for Basic Decision Tree Regressor:", mean_absolute_error(val_y, val_predictions))
df.dtypes
my_model = XGBRegressor()

my_model.fit(train_x, train_y)
predictions = my_model.predict(val_x)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, val_y)))
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)

my_model.fit(train_x, train_y, 

             early_stopping_rounds=5, 

             eval_set=[(val_x, val_y)], 

             verbose=False)
predictionsXBGR = my_model.predict(val_x)

print("Mean Absolute Error using XGBR: " + str(mean_absolute_error(predictionsXBGR, val_y)))