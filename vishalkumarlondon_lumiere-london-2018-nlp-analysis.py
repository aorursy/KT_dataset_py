import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 300) # specifies number of rows to show
pd.options.display.float_format = '{:40,.2f}'.format # specifies default number format to 4 decimal places
pd.options.display.max_colwidth
pd.options.display.max_colwidth = 1000
# This line tells the notebook to show plots inside of the notebook
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sb
import string
import nltk

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sklearn
# Install pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install --upgrade pip
# Install Gensim Library pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install --upgrade gensim
# Install Plotly pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install plotly
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go
from plotly.graph_objs import *

#You can also plot your graphs offline inside a Jupyter Notebook Environment. 
#First you need to initiate the Plotly Notebook mode as below:
init_notebook_mode(connected=True)

#Run at the start of every ipython notebook to use plotly.offline. 
#This injects the plotly.js source files into the notebook.

#https://plot.ly/python/offline/#generating-offline-graphs-within-jupyter-notebook'
#load in twitter data csv file

twitter_data = pd.read_csv('../input/twitter_whole.csv')
twitter_data.head(2)
#import nltk and download all stopwords and punctuations
nltk.download('stopwords')
nltk.download('punkt')

import re
twittertext = twitter_data
#remove 'http' links and remove all '\n' line-brakes

twittertext.Text = twittertext.Text.str.replace(r'http\S+','')
twittertext.Text = twittertext.Text.str.replace(r'\n','')
#remove 'non-ascii' characters

twittertext.Text = twittertext.Text.str.replace(r'[^\x00-\x7F]','')
# twittertext.head(2)
#make all the text lowercase

twittertext.Text = twittertext.Text.str.lower()
#create a column for all #hasthtags that appears in the text
twittertext['hashtags'] = twittertext.Text.str.findall(r'#\S+')
#clean the text by removing all punctuation

# https://stackoverflow.com/questions/23175809/str-translate-gives-typeerror-translate-takes-one-argument-2-given-worked-i

twittertext['cleanText'] = twittertext.Text.apply(lambda x: x.translate(
                                                str.maketrans('','', string.punctuation)))

# twittertext['cleanText'] = twittertext.Text.str.replace(r'[/.!$%^&*():@]','')

twittertext.head(2)
#tokenise all of the clean text

twittertext['tokenText'] = twittertext.cleanText.apply(word_tokenize)
twittertext.head(2)
#count the number of words in the post

twittertext['tokenCount'] = twittertext.tokenText.apply(len)
# twittertext.head(2)
# Import stopwords with nltk.
stop = stopwords.words('english')

# Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
twittertext['noStopWords'] = twittertext['cleanText'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#tokenise all of the text with Stop words

twittertext['tokenTextnoStop'] = twittertext.noStopWords.apply(word_tokenize)
# twittertext.head(2)
twittertext.head(2)
twittertext.shape
#google's langdetect: https://stackoverflow.com/questions/43377265/determine-if-text-is-in-english

#import Google's langdetect library to check where tweets are english or not
from langdetect import detect
lang = []

try:
    for index, row in twittertext['cleanText'].iteritems():
        lang = detect(row)
        twittertext.loc[index, 'Languagereveiw'] = lang

except Exception:
    pass
twittertext.Languagereveiw.value_counts()

#9449 tweets are English
twittertext = twittertext.loc[twittertext['Languagereveiw'] == 'en']
twittertext.shape
twittertext_nodup = twittertext.drop_duplicates(subset='cleanText')
twittertext_nodup.shape
twittertext_nodup.head(2)
# Load in the sentiment analysis csv file

sentiment = pd.read_csv('../input/lumiere-sentiment-nodup.csv')
sentiment.head(1)
sentiment.DateTime.dtypes
sentiment.DateTime = pd.to_datetime(sentiment.DateTime)
sentiment.DateTime.dtypes
sentiment.shape
score = sentiment.score
date_time = sentiment.DateTime

DF = pd.DataFrame()
DF['score'] = score
DF = DF.set_index(date_time)

fig, ax = plt.subplots()
plt.plot(DF)
x = pd.DataFrame(sentiment.groupby('Day')['score'].mean())
x.columns = ['mean']
x['median'] = sentiment.groupby('Day')['score'].median()
x = x.reset_index()
x.head()
people = pd.DataFrame(sentiment.groupby('UserHandle')['score'].mean())
people.columns = ['mean']
people.head()
people['median'] = sentiment.groupby('UserHandle')['score'].median()
people.head()
people['count'] = sentiment.groupby('UserHandle')['score'].count()
people.head()
people = people.reset_index()
people.dtypes
people = people[people['count'] > 4]
people.sort_values(by = ['mean'],
                   ascending = False).head(10)
twittertext_nodup.head(2)
#flatten list of lists of no stop words (TextnoStop) into one large list

tokens = []

for sublist in twittertext_nodup.tokenTextnoStop:
    for word in sublist:
        tokens.append(word)
tokens_df = pd.DataFrame(tokens)
tokens_df.columns = ['words']
tokens_df['freq'] = tokens_df.groupby('words')['words'].transform('count')

tokens_df.shape
tokens_df.head()
tokens_df.words.value_counts()[:20]
word_count = pd.DataFrame(tokens_df.words.value_counts()[:20])
word_count.reset_index(inplace=True)
word_count.columns = ['word', 'count_words']
# twittertext_nodup['hashtags'] = twittertext_nodup.hashtags.str.replace(r'/.!$%^&*():@','')
twittertext_nodup.head(1)
hashtags = []

for sublist in twittertext_nodup.hashtags:
    for word in sublist:
        hashtags.append(word)
hashtags_df = pd.DataFrame(hashtags)
hashtags_df.columns = ['words']
hashtags_df['freq'] = hashtags_df.groupby('words')['words'].transform('count')

hashtags_df.shape
hashtags_df.words.value_counts()[:20]
hash_count = pd.DataFrame(hashtags_df.words.value_counts()[:20])
hash_count.reset_index(inplace=True)
hash_count.columns = ['hashtag', 'count_hashtag']
# hash_count['hashtag'] = hash_count.hashtag.str.replace(r'/.!$%^&*():@','')
count = pd.concat([word_count, hash_count], axis=1)
count
# count.to_csv('lumiere-wordcount.csv')
# words = [w.replace('[br]', '<br />') for w in hashtags]
from nltk.stem import WordNetLemmatizer
from collections import Counter
#import word lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
#group the text by day
day_corpus = twittertext_nodup.groupby('Day').apply(lambda x: x['noStopWords'].str.cat())

#tokenize the text for that day using word lemmatizer and then count
day_corpus = day_corpus.apply(lambda x: Counter([
    wordnet_lemmatizer.lemmatize(w) 
    for w in word_tokenize(x) 
    if w.lower() not in stop and not w.isdigit()]))
#count the five most frequent words per day
day_freq = day_corpus.apply(lambda x: x.most_common(5))

# create a dataframe of the five most frequent words per day
day_freq = pd.DataFrame.from_records(
    day_freq.values.tolist()).set_index(day_freq.index)
day_freq
def normalize_row(x):
    label, repetition = zip(*x)
    t = sum(repetition)
    r = [n/t for n in repetition]
    return list(zip(label,r))

day_freq = day_freq.apply(lambda x: normalize_row(x), axis=1)
twittertext_nodup.head(1)
# create a corpus of all the text which does not have stop words in it
# this corupus will be a list of lists

corpus = list(twittertext_nodup['tokenTextnoStop'])
len(corpus)
from gensim.models import word2vec

model = word2vec.Word2Vec(corpus, 
                          size=20, 
                          window=3, 
                          min_count=40, 
                          workers=10)
model.wv['lumiereldn']

# model.train(corpus, total_examples=len(corpus), epochs=10)
# w1 = 'lumiereldn'
# model.wv.most_similar(positive=w1)
from sklearn.decomposition import PCA

vocab = list(model.wv.vocab)
X = model[model.wv.vocab]

pca = PCA(n_components = 2)
result = pca.fit_transform(X)
import matplotlib.pyplot as pyplot

pyplot.scatter(result[:, 0], result[:, 1])
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i,0], result[i, 1]))
pyplot.show()
wrds = list(model.wv.vocab)
len(wrds)

#zip the two lists containing vectors and words
zipped = zip(model.wv.index2word, model.wv.vectors)

#the resulting list contains `(word, wordvector)` tuples. We can extract the entry for any `word` or `vector` (replace with the word/vector you're looking for) using a list comprehension:
wordresult = [i for i in zipped if i[0] == word]
vecresult = [i for i in zipped if i[1] == vector]
model.wv.most_similar('lumiereldn')
model.wv.most_similar('lumiereldn')
#dataframe of similar words
# https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim

word_frame = pd.DataFrame(result, index=vocab, columns=['x', 'y'])
word_frame.reset_index(inplace=True)
word_frame.columns = ['words', 'x', 'y']
# word_frame.sort_values(by=['x','y'])
nearest = word_frame[word_frame.words.isin(['lumiereldn','amazing', 'open', 'fantastic', 
                                         'mayfair', 'tonight', 'weekend', 'victoria',
                                         'kingscrossn1c', 'wabbey', 
                                         'fitzrovia', 'fabulous', 'trip',
                                         'enjoyed', 'another', 'fun', 'artists'])]
# word_frame.to_csv('lumiere-pca-whole.csv')
# nearest.to_csv('lumiere-nearest-neighbours.csv')
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
%matplotlib inline

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, 
                      n_components=2, 
                      init='pca', 
                      n_iter=1000, 
                      random_state=23, 
                      learning_rate=200)
    
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(50, 50)) 
    
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
tsne_plot(model)

#model
