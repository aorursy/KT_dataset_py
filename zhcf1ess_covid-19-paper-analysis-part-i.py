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

        pass

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import glob

import json

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

import time

import warnings 
def count_ngrams(dataframe, column, begin_ngram, end_ngram):

    # 从 https://stackoverflow.com/questions/36572221/how-to-find-ngram-frequency-of-a-column-in-a-pandas-dataframe 学习

    word_vectorizer = CountVectorizer(ngram_range=(begin_ngram,end_ngram), analyzer='word')

    sparse_matrix = word_vectorizer.fit_transform(df[column].dropna())

    frequencies = sum(sparse_matrix).toarray()[0]

    most_common = pd.DataFrame(frequencies, 

                               index=word_vectorizer.get_feature_names(), 

                               columns=['frequency']).sort_values('frequency',ascending=False)

    most_common['ngram'] = most_common.index

    most_common.reset_index()

    return most_common



def word_cloud_function(df, column, number_of_words):

    # 从 https://www.kaggle.com/benhamner/most-common-forum-topic-words 学习

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    word_string=str(popular_words_nonstop)

    wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=number_of_words,

                          width=1000,height=1000,

                         ).generate(word_string)

    plt.clf()

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()



def word_bar_graph_function(df, column, title, nvals=50):

    # 从 https://www.kaggle.com/benhamner/most-common-forum-topic-words 学习

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    plt.barh(range(nvals), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:nvals])])

    plt.yticks([x + 0.5 for x in range(nvals)], reversed(popular_words_nonstop[0:nvals]))

    plt.title(title)

    plt.show()
# 加载数据

%time

df = pd.read_csv('/kaggle/input/cord19researchchallenge/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
df.head()
df.describe(include='all')
df.journal.value_counts()
df.journal.value_counts()[0:10].plot(kind='bar')

plt.grid()
df.has_full_text.value_counts().plot(kind='bar')

df.publish_time.value_counts()
df.license.value_counts()
plt.figure(figsize=(10,10))

word_bar_graph_function(df,column='title', 

                        title='Most common words in the TITLES of the papers in the CORD-19 dataset',

                        nvals=20)
%time

bi_gram = count_ngrams(df,'title',2,2)
bi_gram[:20]
import plotly.offline as py

py.init_notebook_mode(connected=True)

fig = px.bar(bi_gram.sort_values('frequency',ascending=False)[2:12], 

             x="frequency", 

             y="ngram",

             title='Top Ten relevant bigrams in TITLES of Papers in CORD-19 Dataset',

             orientation='h')

py.iplot(fig)
%time

three_gram = count_ngrams(df,'title',3,3)
three_gram[:20]
fig = px.bar(three_gram.sort_values('frequency',ascending=False)[0:10], 

             x="frequency", 

             y="ngram",

             title='Top Ten 3-Grams in TITLES of Papers in CORD-19 Dataset')

fig.show()
# 词云显示

plt.figure(figsize=(10,10))

word_cloud_function(df,column='title',number_of_words=5000)