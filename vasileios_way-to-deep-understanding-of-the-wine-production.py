import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from bs4 import BeautifulSoup

import re

from nltk.stem import PorterStemmer

from nltk import word_tokenize

from nltk.corpus import stopwords 

from nltk.collocations import *

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

%matplotlib inline
wine = pd.read_csv('../input/winemag-data_first150k.csv', encoding='utf-8')

wine.drop(['Unnamed: 0', 'region_2'], axis = 1, inplace = True)
print('Shape: ', wine.shape)

print('=========================================================================')

print(wine.head())
data = []

for col in wine.columns:

    data.append([col, wine[col].isnull().sum(), '{:.2f}%'.format(float(wine[col].isnull().sum())/len(wine[col])*100) ])

df = pd.DataFrame(data, columns=['columns', 'missing value count', 'missing value percentage'])

df
duplicated_records = wine[wine.duplicated(['country', 'description', 'designation', 'points', 'price', 'province', 'region_1', 'variety', 'winery'])]

duplicated_records.shape
# drop all duplicates and check how many left

new_wine = wine.drop_duplicates()

new_wine.shape
# check again for missing values

data = []

for col in new_wine.columns:

    data.append([col, new_wine[col].isnull().sum(), '{:.2f}%'.format((new_wine[col].isnull().sum())/len(new_wine[col])*100) ])

df = pd.DataFrame(data, columns=['columns', 'missing value count', 'missing value percentage'])

df
# I will remove only the three rows in the dataset that have missing values in the country and province column

new_wine = new_wine.dropna(subset=['country', 'province'])

new_wine.shape
fig, ax = plt.subplots(figsize = (10, 7))

ax = sns.distplot(new_wine['points'], bins = 20, kde=False)
wines100 = new_wine[new_wine['points'] == 100]

wines100['country'].value_counts()
wines100.groupby('country')['price'].mean()
price_by_country_and_province = new_wine.groupby(['country', 'province'])['price'].median()

df_price = pd.DataFrame(price_by_country_and_province)

df_price.reset_index(inplace = True)

df_price.head()
mean_price_in_country = df_price.groupby(['country'])['price'].mean()

df_mean_price = pd.DataFrame(mean_price_in_country)

df_mean_price.reset_index(inplace = True)
fig, ax = plt.subplots(figsize = (15, 8))

ax = sns.barplot(x = 'country', y = 'price', data = df_mean_price)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_title('Distribution of countries whose provinces produce expensive wine');
max_price_in_country = df_price.groupby(['country'])['price'].max()

df_max_price = pd.DataFrame(mean_price_in_country)

df_max_price.reset_index(inplace = True)
fig, ax = plt.subplots(figsize = (15, 8))

ax = sns.barplot(x = 'country', y = 'price', data = df_max_price)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_title('Distribution of countries with expensive provinces');
points_by_country_and_province = new_wine.groupby(['country', 'province'])['points'].mean()

split_points_test = pd.DataFrame(points_by_country_and_province)

split_points_test.reset_index(inplace = True)

split_points_test.head()
t = split_points_test.groupby(['country'])['points'].max()

x = pd.DataFrame(t)

x.reset_index(inplace = True)
fig, ax = plt.subplots(figsize = (15, 8))

ax = sns.barplot(x = 'country', y = 'points', data = x)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
def normalize(text):

    letters = re.sub("[^a-zA-Z]", " ", text)

    words = letters.lower()

    return words
processed_descriptions = []

for description in new_wine['description']:

    norm = normalize(description)

    processed_descriptions.append(norm)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=3)

tfidf_matrix = tfidf_vectorizer.fit_transform(processed_descriptions)
feature_names = tfidf_vectorizer.get_feature_names()

doc = 10

feature_index = tfidf_matrix[:,:].nonzero()[1]

tfidf_scores = zip(feature_index, sorted([tfidf_matrix[doc, x] for x in feature_index]))
# function were taken from here: https://buhrmann.github.io/tfidf-analysis.html

def top_tfidf_feats(row, features, top_n=10):

    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''

    topn_ids = np.argsort(row)[::-1][:top_n]

    top_feats = [(features[i], row[i]) for i in topn_ids]

    df = pd.DataFrame(top_feats)

    df.columns = ['feature', 'tfidf']

    return top_feats



def top_feats_in_doc(Xtr, features, row_id, top_n=10):

    ''' Top tfidf features in specific document (matrix row) '''

    row = np.squeeze(Xtr[row_id].toarray())

    return top_tfidf_feats(row, features, top_n)
# limit dataset to the more popular wine varieties

df_variety = new_wine.groupby('variety').filter(lambda x: len(x) > 5000)
word_scores = []

for i, j in enumerate(df_variety['variety']):

    for item in top_feats_in_doc(tfidf_matrix, feature_names, i):

        word_scores.append(item + (j,))
df = pd.DataFrame(word_scores)

df.columns = ['feature', 'tfidf', 'variety']

df = df.sort_values('tfidf', ascending=False).groupby('variety').head(12)
sns.factorplot(x = 'tfidf', y = 'feature', col='variety', col_wrap=2, data = df, kind='bar', size=10, aspect=.5)
df_country = new_wine.groupby('country').filter(lambda x: len(x) > 3000)
word_scores_by_country = []

for i, j in enumerate(df_country['country']):

    for item in top_feats_in_doc(tfidf_matrix, feature_names, i):

        word_scores_by_country.append(item + (j,))
df = pd.DataFrame(word_scores_by_country)

df.columns = ['feature', 'tfidf', 'country']

df = df.sort_values('tfidf', ascending=False).groupby('country').head(12)
sns.factorplot(x = 'tfidf', y = 'feature', col='country', col_wrap=2, data = df, kind='bar', size=10, aspect=.5)