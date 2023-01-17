# Basic libraries

import numpy as np

import pandas as pd

import warnings

warnings.simplefilter('ignore')

import re



# Directry check

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# datetime

import datetime



# Visualization

import matplotlib.pyplot as plt



# Word stemming

from nltk.stem.porter import PorterStemmer



# Stop word

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words("english")



# Wordcloud

from wordcloud import WordCloud
df_s = pd.read_csv("/kaggle/input/ufo-sightings/scrubbed.csv", header=0)
df_s.head()
# data size

df_s.shape
# null value

df_s.isnull().sum()
# unique values

for i in range(df_s.shape[1]):

    print("-"*50)

    print(df_s.columns[i])

    print(df_s.iloc[:,i].value_counts())
# data info

df_s.info()
df_s.head()
# Create year columns

lists = []

for i in range(len(df_s["datetime"])):

    k = df_s["datetime"][i].split(" ")[0]

    lists.append(k)
df_s["date"] = lists



df_s["dt"] = pd.to_datetime(df_s["date"])

df_s["year"] = df_s["dt"].dt.year
# groupby year

df_year = pd.DataFrame(data=df_s.groupby("year").city.count()).reset_index()

df_year.head()
# visualization

with plt.style.context("fivethirtyeight"):

    plt.figure(figsize=(10,6))

    plt.plot(df_year["year"], df_year["city"])

    plt.xlabel("year")

    plt.ylabel("count")
# groupby city

df_city = pd.DataFrame(data=df_s.groupby("city").country.count()).reset_index().sort_values(by="country", ascending=False)

df_city.head()
with plt.style.context("fivethirtyeight"):

    plt.figure(figsize=(20,6))

    plt.bar(df_city.head(100)["city"], df_city.head(100)["country"])

    plt.xticks(rotation=90)

    plt.xlabel("city")

    plt.ylabel("count")
# groupby city

df_shape = pd.DataFrame(data=df_s.groupby("shape").country.count()).reset_index().sort_values(by="country", ascending=False)

df_shape.head()
with plt.style.context("fivethirtyeight"):

    plt.figure(figsize=(10,6))

    plt.bar(df_shape["shape"], df_shape["country"], color="green")

    plt.xticks(rotation=90)

    plt.xlabel("shape")

    plt.ylabel("count")
# Fill null data of comments

df_s["comments"].fillna("no comment", inplace=True)
text_in = []



for i in range(len(df_s)):

    text = re.sub(r'[^a-zA-Z]', ' ', df_s["comments"][i])

    text = text.lower()

    text = text.split()

               

    # PorterStemmer

    ps = PorterStemmer()

    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]

               

    text_in.extend(text)
# Create word cloud

wordcloud = WordCloud(background_color="black", max_words=300, max_font_size=40, random_state=10).generate(str(text_in))



plt.figure(figsize=(20,12))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
no_light_df = df_s.query("shape!='light'").reset_index()



text_in_2 = []



for i in range(len(no_light_df)):

    text = re.sub(r'[^a-zA-Z]', ' ', no_light_df["comments"][i])

    text = text.lower()

    text = text.split()

               

    # PorterStemmer

    ps = PorterStemmer()

    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]

               

    text_in_2.extend(text)
# Create word cloud

wordcloud = WordCloud(background_color="black", max_words=300, max_font_size=40, random_state=10).generate(str(text_in_2))



plt.figure(figsize=(20,12))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")