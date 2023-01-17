import numpy as np

import pandas as pd

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

from pandas.io.json import json_normalize 

import itertools  # Iterating tools

import re  # Regular Expressions
train_data = pd.read_csv("../input/tmdb-box-office-prediction/train.csv")

test_data = pd.read_csv("../input/tmdb-box-office-prediction/test.csv")
train_data.info()
test_data.info()
train_data.shape , test_data.shape
pd.DataFrame(train_data.isnull().sum()).T
perc_traindata = ((pd.DataFrame(train_data.isnull().sum()).T)/len(train_data))*100

perc_traindata
sns.set(rc={'figure.figsize':(12,8)})

perc_traindata.plot.bar()

plt.xlabel("Columns with missing train data")

plt.ylabel("Percentage of missing train data")
train_data.drop(columns=['belongs_to_collection','homepage'],axis = 1 ,inplace=True)
train_data.drop(columns=['imdb_id', 'poster_path','tagline', 'overview', 'original_title','Keywords' ,'crew'],axis = 1 ,inplace=True)
#Checking again the shape of train data

train_data.shape
train_data.dtypes
#Test data null values

pd.DataFrame(test_data.isnull().sum()).T
perc_testdata = ((pd.DataFrame(test_data.isnull().sum()).T)/len(test_data))*100

perc_testdata
sns.set(rc={'figure.figsize':(12,8)})

perc_testdata.plot.bar()

plt.xlabel("Columns with missing test data")

plt.ylabel("Percentage of missing test data")
test_data.drop(columns=['belongs_to_collection','homepage'],axis = 1 ,inplace=True)

test_data.drop(columns=['imdb_id', 'poster_path','tagline', 'overview', 'original_title','Keywords' ,'crew'],axis = 1 ,inplace=True)
#Checking again the shape of test data

test_data.shape
test_data.dtypes
#extract only genres column from the train dataset and create new dataset of it which contains only genres column

new=train_data.loc[:,["genres"]]

#fill allna with "None"

new["genres"]=train_data["genres"].fillna("None");

new["genres"].head(5)
#extract genre function which will take input as a row [{'id': 35, 'name': 'Comedy'}] and returns

# array of each genre name e.g. ['Comedy'] and if there the row is empty it will return ['None']

def extract_genres(row):

    if row == "None":

        return ['None']

    else:

        results = re.findall(r"'name': '(\w+\s?\w+)'", row)

        return results



#apply extract_genres function on genres column of new dataset    

new["genres"] = new["genres"].apply(extract_genres)

new["genres"].head(10) 
#declare a dictionary 

genres_dict = dict()



# loop through all the rows of genres column and set count of the genre

for genre in new["genres"]:

    for elem in genre:

        if elem not in genres_dict:

            genres_dict[elem] = 1

        else:

            genres_dict[elem] += 1
#generate data from from dictionary which includes count of each genre

genres_df = pd.DataFrame.from_dict(genres_dict, orient='index')
genres_df.columns = ["number_of_movies"]

#sort by number of movies descending 

genres_df = genres_df.sort_values(by="number_of_movies", ascending=False)

#plot bar chart

genres_df.plot.bar()
train_data['spoken_languages']
new=train_data.loc[:,["spoken_languages"]]

#fill allna with "None"

new["spoken_languages"]=train_data["spoken_languages"].fillna("None");

new["spoken_languages"].head(5)
def extract_spoken(row):

    if row == "None":

        return ['None']

    else:

        results = re.findall(r"'name': '(\w+\s?\w+)'", row)

        return results



#apply extract_genres function on genres column of new dataset    

new["spoken_languages"] = new["spoken_languages"].apply(extract_spoken)

new["spoken_languages"].head(10) 
#declare a dictionary 

genres_dict = dict()



# loop through all the rows of genres column and set count of the genre

for genre in new["spoken_languages"]:

    for elem in genre:

        if elem not in genres_dict:

            genres_dict[elem] = 1

        else:

            genres_dict[elem] += 1
#generate data from from dictionary which includes count of each genre

language_df = pd.DataFrame.from_dict(genres_dict, orient='index')

language_df
import time

import datetime



movietime = train_data.loc[:,["title","release_date","budget","runtime","revenue"]]

movietime.dropna()



movietime.release_date = pd.to_datetime(movietime.release_date)

movietime.loc[:,"Year"] = movietime["release_date"].dt.year

movietime.loc[:,"Month"] = movietime["release_date"].dt.month

movietime.loc[:,"Day_of_Week"] = (movietime["release_date"].dt.dayofweek)

movietime.loc[:,"Quarter"]  = movietime.release_date.dt.quarter 



movietime = movietime[movietime.Year<2018]

movietime.head(6)
data_plot = movietime[['revenue', 'Year']]

money_Y = data_plot.groupby('Year')['revenue'].sum()



money_Y.plot(figsize=(15,8))

plt.xlabel("Year of release")

plt.ylabel("revenue")

plt.xticks(np.arange(1970,2020,5))



plt.show()
f,ax = plt.subplots(figsize=(18, 10))

plt.bar(movietime.Month, movietime.revenue, color = 'blue')

plt.xlabel("Month of release")

plt.ylabel("revenue")

plt.show()
x = train_data[['popularity', 'runtime', 'budget']] #independent variables

y= train_data['revenue'] #dependent variable
from sklearn.metrics import mean_squared_log_error as msle

from sklearn.linear_model import LinearRegression



reg = LinearRegression()
from sklearn.model_selection import train_test_split

x, X_test, y, y_test = train_test_split(x, y, test_size=0.30, random_state=1)
x.describe()
x.isnull().sum()
#we fill the null values in the train data

x.median()
medianFiller = lambda t: t.fillna(t.median())

x = x.apply(medianFiller,axis=0)
x.isnull().sum()
model = reg.fit(x,y)

model
y_pred = reg.predict(x)

for idx, col_name in enumerate(x.columns):

    print("The coefficient for {} is {}".format(col_name, model.coef_[0]))
intercept = model.intercept_

print("The intercept for our model is {}".format(intercept))
#The score (R^2) for in-sample and out of sample

model.score(x, y)


x_test = test_data[['popularity', 'runtime', 'budget']]

x_test.isnull().sum()

#pred = reg.predict(x_test)
x_test.median()
medianFiller = lambda t: t.fillna(t.median())

x_test = x_test.apply(medianFiller,axis=0)
pred = reg.predict(x_test)

pred