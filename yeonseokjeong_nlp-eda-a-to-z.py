import torch

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
### 1> set several paths

PATH_TRAIN = '../input/nlp-getting-started/train.csv'

PATH_TEST = '../input/nlp-getting-started/test.csv'



### 2> read_csv

df_train = pd.read_csv(PATH_TRAIN)

df_test = pd.read_csv(PATH_TEST)
df_train.head()
df_train.info()
df_train.target.value_counts()
df_train.loc[df_train['target'] == 1, 'text'].str.split()
def make_list_by_target(target):

    corpus = []

    

    for x in df_train.loc[df_train['target'] == target, 'text'].str.split():

        # x에는 각 idx별 text_data가 들어간다.

        for i in x:

            # i에는 각 text_data별 단어들이 들어간다. 

            corpus.append(i)

            

    return corpus
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from collections import defaultdict
corpus = make_list_by_target(0)



dic = defaultdict(int)



for word in corpus:

    if word in stop:

        dic[word] += 1

        

top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:10]



x, y = zip(*top)

plt.title('With no disaster')

plt.bar(x, y, color = 'red')
corpus = make_list_by_target(1)



dic = defaultdict(int)



for word in corpus:

    if word in stop:

        dic[word] += 1

        

top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:10]



x, y = zip(*top)



plt.title('With disaster')

plt.bar(x, y, color='green')
import string
plt.figure(figsize=(10, 5))

corpus = make_list_by_target(0)



dic = defaultdict(int)



special = string.punctuation

for i in corpus:

    if i in special:

        dic[i] += 1

        

x, y = zip(*dic.items())

plt.bar(x, y, color = 'red')
plt.figure(figsize=(10, 5))

corpus = make_list_by_target(1)



dic = defaultdict(int)



special = string.punctuation

for i in corpus:

    if i in special:

        dic[i] += 1

        

x, y = zip(*dic.items())

plt.bar(x, y, color='green')
from collections import Counter
counter = Counter(make_list_by_target(0))

most_common = counter.most_common()



x = list()

y = list()



for word, count in most_common[:40]:

    if word not in stop:

        x.append(word)

        y.append(count)

        

sns.barplot(x=y, y=x, saturation=1)
counter = Counter(make_list_by_target(1))

most_common = counter.most_common()



x = list()

y = list()



for word, count in most_common[:40]:

    if word not in stop:

        x.append(word)

        y.append(count)

        

sns.barplot(x=y, y=x, saturation=1)
from sklearn.feature_extraction.text import CountVectorizer
def get_top_tweet_bigrams(corpus, n=10):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0)

    

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    

    return words_freq[:n]
plt.figure(figsize=(10, 5))

top_tweet_bigrams = get_top_tweet_bigrams(df_train['text'])[:10]



x, y = map(list, zip(*top_tweet_bigrams))



sns.barplot(x=y, y=x)