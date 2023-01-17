import os 

import json 

from pprint import pprint 

from copy import deepcopy 

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import plotly.express as px 

import plotly.graph_objects as go 

import plotly.offline as py 

from wordcloud import WordCloud, STOPWORDS 

from collections import Counter 

from nltk.corpus import stopwords 

from sklearn.feature_extraction.text import CountVectorizer 

import warnings 

warnings.filterwarnings('ignore') 

df = pd.read_csv('../input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv') 

df1 = pd.read_csv('../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv')

# make new dataframe for journal informations

dff = df.copy()

dff['WHO #Covidence'] = dff['WHO #Covidence'].str.replace('#','') 

dff['WHO #Covidence'] = dff['WHO #Covidence'].astype("Float32").astype("Int32")

journal_unique = dff['journal'].value_counts().rename_axis('journal').reset_index(name='counts') 

covidence_mean = dff.groupby('journal').mean().reset_index()

df_journal = pd.merge(journal_unique, covidence_mean, on="journal")

df_journal.drop(["Microsoft Academic Paper ID","pubmed_id"], axis = 1, inplace = True) 

fig = px.bar(df_journal[:20], x='counts', y='journal', 

             title='Most Common Journals in the CORD-19 Dataset', orientation='h') 

fig.show() 

temp = df_journal.sort_values(by=['WHO #Covidence'], ascending=False)

fig = px.bar(temp[:20], x='WHO #Covidence', y='journal', 

             title='Highest WHO #Covidence of Journal', orientation='h') 

fig.show() 
fig = go.Figure(data=go.Scatter(x=df_journal['has_full_text'], y=df_journal['journal'], 

                                mode='markers'))

fig.update_layout(title_text = 'Distribution of "has_full_text" among journals')

fig.update_layout(

    xaxis=dict(autorange=True), 

    yaxis=dict(autorange=True, showticklabels=False)

)

fig.show()                
def count_ngrams(dataframe,column,begin_ngram,end_ngram):

    # adapted from https://stackoverflow.com/questions/36572221/how-to-find-ngram-frequency-of-a-column-in-a-pandas-dataframe

    word_vectorizer = CountVectorizer(ngram_range=(begin_ngram,end_ngram), analyzer='word')

    sparse_matrix = word_vectorizer.fit_transform(df1['title'].dropna())

    frequencies = sum(sparse_matrix).toarray()[0]

    most_common = pd.DataFrame(frequencies, 

                               index=word_vectorizer.get_feature_names(), 

                               columns=['frequency']).sort_values('frequency',ascending=False)

    most_common['ngram'] = most_common.index

    most_common.reset_index()

    return most_common



def word_cloud_function(df,column,number_of_words):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df1[column] if isinstance(x, str)]

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



def word_bar_graph_function(df,column,title):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])

    plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))

    plt.title(title)

    plt.show()

    

three_gram = count_ngrams(df1,'title',3,3)

words_to_exclude = ["my","to","at","for","it","the","with","from","would","there","or","if","it","but","of","in","as","and",'NaN','dtype']

plt.figure(figsize=(10,10))

word_cloud_function(df1,'title',50000)
fig = px.bar(three_gram.sort_values('frequency',ascending=False)[0:10], x="frequency", y="ngram",

             title='Most Common 3-Words in Titles of Papers', orientation='h') 

fig.show()