import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from collections import  Counter

plt.style.use('ggplot')

stop=set(stopwords.words('english'))

import re

from nltk.tokenize import word_tokenize

import gensim

import string

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots

import plotly.figure_factory as ff

train = pd.read_csv('../input/nlp-getting-started/train.csv')

test  = pd.read_csv('../input/nlp-getting-started/test.csv')

train.head()
print('There are {} rows and {} columns in the train data'.format(train.shape[0],train.shape[1]))

print('There are {} rows and {} columns in the test data'.format(test.shape[0],test.shape[1]))
print('Number of real disaster tweets {} , {} %'.format(train[train.target==1].shape[0],train[train.target==1].shape[0]/train.shape[0] *100))

print('Number of fake disaster tweets {} , {} %'.format(train[train.target==0].shape[0],train[train.target==0].shape[0]/train.shape[0] *100))
tweet_len_disaster     =train[train['target']==1]['text'].str.len()

tweet_len_non_disaster =train[train['target']==0]['text'].str.len()
fig = make_subplots(rows=1, cols=2, subplot_titles=('Disaster',"Non-disaster"))



trace0= go.Histogram(

    

    x=tweet_len_disaster,

    name="Disaster",

    opacity=0.75

)



trace1= go.Histogram(

    

    x=tweet_len_non_disaster,

    name="Non-Disaster",

    opacity=0.75

)



fig.append_trace(trace0,1,1)

fig.append_trace(trace1,1,2)



fig.update_layout(template="plotly_dark",title_text='<b>Distribution of length of characters in tweets</b>',font=dict(family="Arial,Balto,Courier new,Droid sans",color='white'))

fig.show()
tweet_len_1=train[train['target']==1]['text'].str.split().map(lambda x: len(x))

tweet_len_0=train[train['target']==0]['text'].str.split().map(lambda x: len(x))
fig = make_subplots(rows=1, cols=2, subplot_titles=('Disaster',"Non-disaster"))



trace0= go.Histogram(

    

    x=tweet_len_1,

    name="Disaster",

    opacity=0.75

)



trace1= go.Histogram(

    

    x=tweet_len_0,

    name="Non-Disaster",

    opacity=0.75

)



fig.append_trace(trace0,1,1)

fig.append_trace(trace1,1,2)



fig.update_layout(template="plotly_dark",title_text='<b>Distribution of length of words in tweets</b>',font=dict(family="Arial,Balto,Courier new,Droid sans",color='white'))

fig.show()
word_1=train[train['target']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])

word_0=train[train['target']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
word_1=word_1.map(lambda x: np.mean(x))

word_0=word_0.map(lambda x: np.mean(x))
hist_data = [word_1,word_0]

group_labels = ['disaster','non-disaster']

fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

fig.update_layout(title_text='Average word length in a tweet',width=900,height=450)

fig.show()
def create_corpus(target):

    corpus=[]

    

    for x in train[train['target']==target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus
corpus_0=create_corpus(0)



dic_0=defaultdict(int)

for word in corpus_0:

    if word in stop:

        dic_0[word]+=1



top_0=sorted(dic_0.items(), key=lambda x:x[1],reverse=True)[:10] 

    





x_0,y_0=zip(*top_0)





corpus_1=create_corpus(1)



dic_1=defaultdict(int)

for word in corpus_1:

    if word in stop:

        dic_1[word]+=1

        

top_1=sorted(dic_1.items(), key=lambda x:x[1],reverse=True)[:10] 



x_1,y_1=zip(*top_1)

fig = go.Figure(data=[

    go.Bar(name='non-disaster', x=x_0, y=y_0),

    go.Bar(name='diaster', x=x_1, y=y_1)

])

fig.update_layout(title_text='common stop words')

fig.show()

import string

special = string.punctuation







corpus0=create_corpus(0)



dic0=defaultdict(int)



for i in (corpus0):

    if i in special:

        dic0[i]+=1

        



corpus1=create_corpus(1)



dic1=defaultdict(int)

for i in (corpus1):

    if i in special:

        dic1[i]+=1

      



    

x0,y0=zip(*dic0.items())



x1,y1=zip(*dic1.items())    

        



    

    

fig = go.Figure(data=[

    go.Bar(name='non-disaster', x=x0, y=y0),

    go.Bar(name='diaster', x=x1, y=y1)

])

fig.update_layout(title_text='Punctuations')

fig.show()

    
counter0=Counter(corpus0)

most0=counter0.most_common()

x0=[]

y0=[]

for word,count in most0[:100]:

    if (word not in stop) :

        x0.append(word)

        y0.append(count)



counter1=Counter(corpus1)

most1=counter1.most_common()

x1=[]

y1=[]

for word,count in most1[:100]:

    if (word not in stop) :

        x1.append(word)

        y1.append(count)

fig = go.Figure(data=[

    go.Bar(name='non-disaster', x=x0, y=y0),

    go.Bar(name='diaster', x=x1, y=y1)

])

fig.update_layout(title_text='common  words')

fig.show()
def get_top_tweet_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
top_tweet_bigrams=get_top_tweet_bigrams(train[train.target==0]['text'])[:10]

x0,y0=map(list,zip(*top_tweet_bigrams))

top_tweet_bigrams=get_top_tweet_bigrams(train[train.target==1]['text'])[:10]

x1,y1=map(list,zip(*top_tweet_bigrams))
fig = go.Figure(data=[

    go.Bar(name='non-disaster', x=x0, y=y0),

    go.Bar(name='diaster', x=x1, y=y1)

])

fig.show()
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)
remove_URL("New competition launched :https://www.kaggle.com/c/nlp-getting-started")
train['text']=train['text'].apply(lambda x : remove_URL(x))

def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)
example = """<div>

<h1>Real or Fake</h1>

<p>Kaggle </p>

<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>

</div>"""

print(remove_html(example))
train['text']=train['text'].apply(lambda x : remove_html(x))
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)
remove_emoji("Omg another Earthquake 😔😔")
train['text']=train['text'].apply(lambda x: remove_emoji(x))
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



example="I am a #king"

print(remove_punct(example))

train['text']=train['text'].apply(lambda x : remove_punct(x))