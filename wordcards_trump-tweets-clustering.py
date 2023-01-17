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
import warnings; warnings.simplefilter('ignore')

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

import re

from wordcloud import WordCloud

import textwrap

from sklearn.cluster import KMeans

from absl import logging

import tensorflow_hub as hub

# Embed with Universal Sentence Encoder

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
data=pd.read_csv('../input/trump-tweets/trumptweets.csv').set_index('id')

data.drop(['link','mentions','hashtags','geo'], axis=1, inplace=True)

data.head()
# removing URLs

url_pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

pic_pattern = re.compile('pic\.twitter\.com/.{10}')

data['text'] = data['content'].apply(lambda buf: url_pattern.sub('',buf))

data['text'] = data['text'].apply(lambda buf: pic_pattern.sub('', buf))



# removing space in mentions and hashtags

data['text'] = data['text'].apply(lambda buf: buf.replace('@ ', '@'))

data['text'] = data['text'].apply(lambda buf: buf.replace('# ', '#'))

data['year'] = data['date'].apply(lambda buf: int(buf[:4]))
fig, ax = plt.subplots(3,1,figsize=(12,12))

plt.subplots_adjust(hspace=0.3)

sns.countplot(x='year', data=data, ax=ax[0])

sns.barplot(x='year', y='favorites', data=data, ax=ax[1])

sns.barplot(x='year', y='retweets', data=data, ax=ax[2]);
data_president = data[data['year'] > 2016]

data_pre_president = data[data['year'] < 2017]

def wc_president(president):

    if president:

        tmp_data = data_president

        title = 'As President (2017-)'

    else:

        tmp_data = data_pre_president

        title = 'Before President (-2016)'

    words = ' '.join([text for text in tmp_data['text']])

    wordcloud = WordCloud(width=800, height=400, background_color='white', max_font_size=110).generate(words)

    plt.figure(figsize=(16, 8))

    plt.title(title, fontsize=32)

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis('off');
wc_president(False)

wc_president(True)
# Eliminate too short tweets in order to clustering make sence

data_president['len_word'] = data_president['text'].str.split().map(lambda x:len(x))

data_president2 = data_president[data_president['len_word']>1]



# text embedding with Universal Sentence Encoder 

X = embed(data_president2['text'].values)



# Clustering

kmeans = KMeans(n_clusters=4, random_state=42).fit(X)



# Assign cluster number to each text

data_president2= pd.concat([data_president2,

                            pd.DataFrame(kmeans.labels_, index=data_president2.index).rename(columns={0:'Cluster'})],

                           axis=1)



# Assign Cosine Similality to each Cluster Center

def cos_sim(v1, v2):

    return np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))



norm = np.array(np.zeros(len(X)*4).reshape(len(X), 4))

for i in range(len(X)):

    for j in range(4):

        norm[i, j] = cos_sim(X[i], kmeans.cluster_centers_[j])



grp_df = pd.concat([data_president2,

                    pd.DataFrame(norm, index=data_president2.index).rename(columns={0:'sim0', 1:'sim1', 2:'sim2', 3:'sim3'})],

                    axis=1)



# Divide tweets data by Cluster

df_0 = grp_df[grp_df['Cluster']==0].sort_values('sim0', ascending=False)

df_1 = grp_df[grp_df['Cluster']==1].sort_values('sim1', ascending=False)

df_2 = grp_df[grp_df['Cluster']==2].sort_values('sim2', ascending=False)

df_3 = grp_df[grp_df['Cluster']==3].sort_values('sim3', ascending=False)
def wcshow(df, title):

    words = ' '.join([text for text in df['text']])

    wordcloud = WordCloud(width=800, height=400, background_color='white', max_font_size=110).generate(words)

    plt.figure(figsize=(16, 8))

    plt.title(title, fontsize=32)

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis('off');



def typical_tweets(df):

    for i in range(5):

        wrap_list = textwrap.wrap(df.iloc[i,4], 76)

        print('\n'.join(wrap_list),'\n')
wcshow(df_0, 'Cluster: 0')

wcshow(df_1, 'Cluster: 1')

wcshow(df_2, 'Cluster: 2')

wcshow(df_3, 'Cluster: 3')
typical_tweets(df_0)
typical_tweets(df_1)
typical_tweets(df_2)
typical_tweets(df_3)
fig, ax = plt.subplots(3, 1, figsize=(9,15))

plt.subplots_adjust(hspace=0.3)

sns.barplot(x='Cluster', y='len_word', data=grp_df, ax=ax[0])

sns.barplot(x='Cluster', y='retweets', data=grp_df, ax=ax[1])

sns.barplot(x='Cluster', y='favorites', data=grp_df, ax=ax[2]);