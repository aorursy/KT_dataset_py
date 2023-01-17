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
# Imports

from pandas.io.json import json_normalize

import  nltk, urllib.error, urllib.parse, urllib.request, json, datetime

import matplotlib.pyplot as plt



import plotly.express as px

import plotly.graph_objects as go 

from plotly.subplots import make_subplots





# News sources' ids from NewsAPI.org

abc_news = 'abc-news'

cnn = 'cnn'

fox_news = 'fox-news'

usa_today = 'usa-today'

politico = 'politico'

washington_post = 'the-washington-post'

reuters = 'reuters'





# Date formula (found online)

def monthAgoDate():

    today = datetime.datetime.today()

    if today.month == 1:

        one_month_ago = today.replace(year=today.year - 1, month=12)

    else:

        extra_days = 0

        while True:

            try:

                one_month_ago = today.replace(month=today.month - 1, day=today.day - extra_days)

                break

            except ValueError:

                extra_days += 1

    return one_month_ago





# NewsAPI.org secret key

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("NewsAPI.org secret key")





# NewAPI.org code to source news

def getStories(query,date,sources):

        key = secret_value_0

        url = "http://newsapi.org/v2/everything?qInTitle="+query+"&sources="+sources+"&pageSize=100&apiKey="+key

        return urllib.request.urlopen(url)
# Getting news Trump stories from different publications

trump_abc = json.load(getStories('Trump',monthAgoDate(), abc_news))

trump_cnn = json.load(getStories('Trump',monthAgoDate(), cnn))

trump_fox = json.load(getStories('Trump',monthAgoDate(), fox_news))

trump_usa = json.load(getStories('Trump',monthAgoDate(), usa_today))

trump_politico = json.load(getStories('Trump',monthAgoDate(), politico))

trump_washington_post = json.load(getStories('Trump',monthAgoDate(), washington_post))

trump_reuters = json.load(getStories('Trump',monthAgoDate(), reuters))
# Creating dataframes for each news source

df_trump_abc = pd.DataFrame()

df_trump_cnn = pd.DataFrame()

df_trump_fox = pd.DataFrame()

df_trump_usa = pd.DataFrame()

df_trump_politico = pd.DataFrame()

df_trump_washington_post = pd.DataFrame()

df_trump_reuters = pd.DataFrame()
# Populating dataframes with news stories' data arranged into titles, descriptions, publishing date and news source. 

# Then dropping any NAs

def serializeFrames(dataframe, data, source):

    dataframe = json_normalize(data['articles'])

    dataframe = dataframe[['title','description','publishedAt','source.name']].dropna()

    return dataframe
# Adding data from each news source

df_trump_abc = serializeFrames(df_trump_abc, trump_abc, 'ABC')

df_trump_cnn = serializeFrames(df_trump_cnn, trump_cnn, 'CNN')

df_trump_fox = serializeFrames(df_trump_fox, trump_fox, 'Fox News')

df_trump_usa = serializeFrames(df_trump_usa, trump_usa, 'USA Today')

df_trump_politico = serializeFrames(df_trump_politico, trump_politico, 'Politico')

df_trump_washington_post = serializeFrames(df_trump_washington_post, trump_washington_post, 'The Washington Post')

df_trump_reuters = serializeFrames(df_trump_reuters, trump_reuters, 'Reuters')
# Combining individual dataframes into one dataframe

df = pd.concat([df_trump_abc, df_trump_cnn, df_trump_fox, df_trump_usa, df_trump_politico, df_trump_washington_post, df_trump_reuters], ignore_index=True)

df
# Applying Vader's sentiment analysis

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()





# Code found online. 0.05 is a threshold for statistical significance 

def sentiment_analysis(sentence):

    sentiment = sia.polarity_scores(sentence)

    if sentiment['compound'] >= 0.05:

        return "Positive"

    elif sentiment['compound'] <= -0.05:

        return "Negative"

    else:

        return "Neutral"



    

def trump_sentiment_analysis(data):

    data['title_sentiment'] = data['title'].apply(lambda sentence: sentiment_analysis(sentence))

    data['description_sentiment'] = data['description'].apply(lambda sentence: sentiment_analysis(sentence))



trump_sentiment_analysis(df)

df
# Saving the result into the csv file

df.to_csv(r'trump_news1.csv')
# Title sentiments from MonkeyLearn

df_title_monkeylearn = pd.read_csv("../input/MonkeyLearn_title.csv").dropna()

df_title_monkeylearn
# Description sentiments from MonkeyLearn

df_description_monkeylearn = pd.read_csv("../input/MonkeyLearn_description.csv").dropna()

df_description_monkeylearn
# Title sentiments from Vader

titles_Vader = df.groupby(['title_sentiment']).size()

plt.title('Title Sentiment (Vader)')

plt.ylabel('Count')

titles_Vader.plot(kind='bar',figsize=(10, 5))
# Title sentiments from MonkeyLearn

titles_MonkeyLearn = df_title_monkeylearn.groupby(['Classification']).size()

plt.title('Title Sentiment (MonkeyLearn)')

plt.ylabel('Count')

titles_MonkeyLearn.plot(kind='bar',figsize=(10, 5))
descriptions_Vader = df.groupby(['description_sentiment']).size()

plt.title('Description Sentiment (Vader)')

plt.ylabel('Count')

descriptions_Vader.plot(kind='bar',figsize=(10, 5))
descriptions_MonkeyLearn = df_description_monkeylearn.groupby(['Classification']).size()

plt.title('Description Sentiment (MonkeyLearn)')

plt.ylabel('Count')

descriptions_MonkeyLearn.plot(kind='bar',figsize=(10, 5))
title_by_source_Vader = df.groupby(['source.name','title_sentiment']).size().unstack().plot(kind='bar',figsize=(10, 5), stacked=True)

plt.title('Title Sentiment by Source (Vader)')

plt.ylabel('Count')

plt.show()
title_by_source_MonkeyLearn = df_title_monkeylearn.groupby(['source.name','Classification']).size().unstack().plot(kind='bar',figsize=(10, 5), stacked=True)

plt.title('Title Sentiment by Source (MonkeyLearn)')

plt.ylabel('Count')

plt.show()