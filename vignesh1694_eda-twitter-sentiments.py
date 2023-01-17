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
import pandas as pd
import numpy as np
import pyodbc
from pandas import DataFrame

import string
import nltk
from sklearn import re
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist 
from nltk.stem.porter import *
import matplotlib.pyplot as plt 
lemma = WordNetLemmatizer()

from fuzzywuzzy import process
from fuzzywuzzy import fuzz

import matplotlib.pyplot as plt
import seaborn as sns

import time
import csv
import datetime
from datetime import datetime
from datetime import date, timedelta
import os, re


# vectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression #classification model
from sklearn.metrics import confusion_matrix, classification_report, f1_score 


df_train = pd.read_csv("../input/twitter-sentiment-analysis-hatred-speech/train.csv")
df_test = pd.read_csv("../input/twitter-sentiment-analysis-hatred-speech/test.csv")
df_train
print(len(df_train[df_train.label == 0]), 'Non-Hatred Tweets')
print(len(df_train[df_train.label == 1]), 'Hatred Tweets')
df_combined = df_train.append(df_test, ignore_index=True, sort = False)
df_combined
def remove_pattern(input_text, pattern):
    r = re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text)
        
    return input_text   
# remove twitter handles (@user)
df_combined['tidy_tweet'] = np.vectorize(remove_pattern)(df_combined['tweet'], "@[\w]*")
df_combined
# remove special characters, numbers, punctuations
df_combined['tidy_tweet'] = df_combined['tidy_tweet'].str.replace('[^\w\d#\s]',' ')
df_combined
df_combined['tidy_tweet'] = df_combined['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
df_combined
tokenized_tweet = df_combined['tidy_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

df_combined['tidy_tweet_stemmed'] = tokenized_tweet
df_combined

def extract_hashtag(tweet):
    tweets = " ".join(filter(lambda x: x[0]== '#', tweet.split()))
    tweets = re.sub('[^a-zA-Z]',' ',  tweets)
    tweets = tweets.lower()
    tweets = [lemma.lemmatize(word) for word in tweets]
    tweets = "".join(tweets)
    return tweets

df_combined['hashtag'] = df_combined.tweet.apply(extract_hashtag)
df_combined
df_hashtag = df_combined['hashtag'].apply(lambda x: x.split())
df_hashtag = df_hashtag.to_list()
df_hashtag
hash_list = [j for i in df_hashtag for j in i]
hash_list = pd.DataFrame(hash_list)
hash_list = hash_list.rename(columns = {0 : 'hashtags'})
hash_list
hash_list = hash_list.hashtags.str.split(expand=True).stack().value_counts()
hash_list = pd.DataFrame(hash_list)
hash_list = hash_list.reset_index()
hash_list = hash_list.rename(columns = {'index' : 'hashtags', 0 : 'count'})
hash_list
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

x_pos = hash_list['hashtags']
y_pos = hash_list['count']

plt.bar(y_pos, x_pos, align='center', alpha=0.5)
plt.xticks(y_pos, x_pos)
plt.ylabel('Count')
plt.title('HashTags')

plt.show()


import plotly.graph_objects as go

fig = go.Figure(go.Bar(
            y=hash_list['hashtags'],
            x=hash_list['count'],
            orientation='h'))

fig.show()
