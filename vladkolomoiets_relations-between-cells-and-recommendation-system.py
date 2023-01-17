# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import missingno as miss



%matplotlib inline



df = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv', index_col=0)

df.head()
df.head(10)
df.info()
df.isnull().sum()
miss.matrix(df)

plt.show()
miss.heatmap(df)

plt.show()
df.groupby(['region_2']).mean()
region_group = df.groupby('region_2').mean().reset_index('region_2')

regions = region_group['region_2']

price = region_group['price']

points = region_group['points']



fig, ax1 = plt.subplots()



ax2 = ax1.twinx()

ax1.bar(regions, price)

ax2.plot(regions, points, color='green')



ax1.set_xlabel('Region')

ax1.set_ylabel('Price ($)', color='g')

ax2.set_ylabel('Points', color='green')

ax1.set_xticklabels(regions, rotation='vertical', size=8)



plt.show()
def get_year(title):

    title = title.split()

    for ele in title:

        if ele.isnumeric() and 1920 <= int(ele) <= 2020:

            return ele





df['year'] = df['title'].apply(get_year) 
df.head()
df.groupby(['year']).mean().astype(int).tail(10)
year_group = df.groupby('year').mean().reset_index('year')

years = year_group['year']

price = year_group['price']

points = year_group['points']



fig, ax1 = plt.subplots(figsize=(50, 25))



ax2 = ax1.twinx()

ax1.bar(years, price)

ax2.plot(years, points, color='green')



label_format = '{:.0f}'



ax1.set_ylabel('Price ($)', size=30)

ax1.set_xlabel('Years', size=30)

ax2.set_ylabel('Points', size=30, color='g')



ax1.set_xticklabels(years, size=25, rotation=45)

ax1.set_yticks(ax1.get_yticks().tolist())

ax1.set_yticklabels([label_format.format(x) for x in ax1.get_yticks().tolist()], fontsize=30)

ax2.set_yticks(ax2.get_yticks().tolist())

ax2.set_yticklabels([label_format.format(x) for x in ax2.get_yticks().tolist()], fontsize=30)

plt.show() 

df['description'] = df['description'].apply(lambda string: string.replace(',', ' '))

df['keywords'] = df['description'].apply(lambda string: string.split())

df['keywords'] = df['keywords'].map(lambda x: list(map(str.lower, x)))



import nltk

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords



tokenizer = nltk.tokenize.WhitespaceTokenizer()

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))



def lemmatize_text(text):

    return [lemmatizer.lemmatize(w) for w in text if w not in stop_words and w.isalpha()]



df['keywords'] = df['keywords'].apply(lemmatize_text)
df.head()
def recommendation(title):

    wine_column = df.loc[df['title'] == title]

    index = df.loc[df['title'] == title].index.values.astype(int)[0] 

    df['points'] = [len(list(set(df['keywords'][index]) & set(df['keywords'][i]))) for i in range(len(df))]

    return(df.sort_values(by='points', ascending=False).drop(['taster_name', 'taster_twitter_handle'], axis=1).iloc[1:].head(5))



recommendation('Béres 2014 Furmint (Tokaj)')