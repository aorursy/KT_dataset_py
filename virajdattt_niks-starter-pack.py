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
#This is a python library for plotting 

import seaborn as sns



#from html2text import unescape

#The data for this competetion is given in a csv format 



#read_csv helps us load the data as a data frame

raw_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv",encoding="utf-8")

raw_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv", encoding="utf-8")

#shape property is for seeing number of rows vs columns

#columns are also known as features and rows as samples

print("The size of Train set",raw_train.shape)

print("The size of Test set",raw_test.shape)
#This is to print the data type of each column we have

raw_train.dtypes
#to return top n (5 by default) rows of a data frame or series.

raw_train.head()
#value_counts method lists the number of occurence of a category in a column

# When we run this we see USA 104, which means that the word USA has appeared 104 times in the 

# location column(feature)

raw_train['location'].value_counts()
#This is to list the number of rows with any column without data/NAN



raw_train.isnull().sum()
raw_train['target'].value_counts()
sns.countplot(raw_train['target'])
#Taking backup of the original dataframe

train = raw_train
#let us extract the number of words in each tweet

train['word_count'] = train['text'].apply(lambda x: len(str(x).split(" ")))

#print(train['word_count'])
train.head()
print("Average word count for all tweets", train['word_count'].mean())

print("Average word count for non-disaster tweet", train[train['target']==0]['word_count'].mean())

print("Average word count for disaster tweet", train[train['target']==1]['word_count'].mean())
sns.countplot(train[train['target']==0]['word_count'])
sns.countplot(train[train['target']==1]['word_count'])
train['hastags'] = train['text'].apply(lambda x: [ x for x in x.split() if x.startswith("#")])

train['hastags'] = train['hastags'].apply(lambda x: ','.join(map(str, x)))
train.head()
from wordcloud import WordCloud

import matplotlib.pyplot as plt

#Building the word cloud for MOST used WORDS

print("The MOST used WORDS:")

wordcloud = WordCloud(width = 1000, height = 500,background_color='white').generate(' '.join(train['text']))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
from collections import Counter

rslt = pd.DataFrame(Counter(" ".join(train["text"]).split()).most_common(20),columns=['Word', 'Frequency']).set_index('Word')

rslt.plot.bar(rot=0, figsize=(16,10), width=0.8)

plt.show()
#Extract captial letters from the tweets

train['capitals'] = train['text'].str.extract(r'([A-Z]+(?:(?!\s?[A-Z][a-z])\s?[A-Z])+)')
train.head()
train['hastags'].isnull().sum()
train['capitals'].isnull().sum()
#train[train['keyword']=='aftershock']['target']
for i in train['keyword'].unique():

        #dis = len(train[train[train['keyword']==i]['target'] ==1])

        #ndis = len(train[train[train['keyword']==i]['target'] ==0])

        #total = dis+ndis

        print("The probablity of disaster with key word-->",i,"-->", train[train['keyword']==i]['target'].mean())

        #print("For",i,":--",train[train['keyword']==i]['target'].value_counts())
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

from nltk.corpus import stopwords 

import re

#Clean the text by retaining only alphabets and converting all the characters to small

def clean_up(review):

    clean = re.sub("[^a-zA-Z]"," ",review)

    #replace multiple space by a single

    clean = re.sub(' +', ' ',clean)

    

    word_tokens= clean.lower().split()

    

    # 4. Remove stopwords

    le=WordNetLemmatizer()

    stop_words= set(stopwords.words("english")) 

    stop_words.add("co")

    stop_words.add("http")

    word_tokens= [le.lemmatize(w) for w in word_tokens if not w in stop_words]

    

    cleaned =" ".join(word_tokens)

    #re.sub(' +', ' ',string4)

    return cleaned
train['cleaned'] = train['text'].apply(clean_up)
train.head()
from wordcloud import WordCloud

import matplotlib.pyplot as plt

#Building the word cloud for MOST used WORDS

print("The MOST used WORDS:")

wordcloud = WordCloud(width = 1000, height = 500,background_color='white').generate(' '.join(train['cleaned']))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
from collections import Counter

rslt = pd.DataFrame(Counter(" ".join(train["cleaned"]).split()).most_common(20),columns=['Word', 'Frequency']).set_index('Word')

rslt.plot.bar(rot=0, figsize=(16,10), width=0.8)

plt.show()
train.head()
train[train['hastags']!='']['target'].mean()
len(train[train['capitals'].notnull()])
train[train['capitals'].notnull()]['target'].mean()
vocab = train['cleaned'].values
vocab = " ".join(vocab).split()
unique_words = (set(" ".join(vocab).split()))
print("The number of words in all tweets -->", len(vocab))

print("The number of unique words in all tweets -->", len(unique_words))
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn import svm

text_clf = Pipeline([

('vect', CountVectorizer()),

('tfidf', TfidfTransformer()),

('clf', svm.LinearSVC()),

])
text_clf.fit(train['cleaned'], train['target'])
text_clf.score(train['cleaned'], train['target'])
raw_test['cleaned'] = raw_test['text'].apply(clean_up)
text_clf.predict(raw_test['cleaned'])
sub_file = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv", encoding="utf-8")
sub_file.head()
sub_file['target'] = text_clf.predict(raw_test['cleaned'])

sub_file.to_csv("submission.csv", index = False)
#train[train['hastags']!='']['hastags'].value_counts()
#raw_train['keyword'].value_counts().index.tolist()
#raw_train['keyword'].str.replace('%20', ' ').value_counts().index.tolist()
#train['location'].value_counts()
#raw_test['location'].value_counts()
# from textblob import TextBlob

# train['location'].head(5).apply(lambda txt: ''.join(TextBlob(str(txt)).correct()))

#train[train['location']=='Thrissur']['target'].mean()
#Niks
#stemming text column

# import these modules 

from nltk.stem import PorterStemmer 

from nltk.tokenize import word_tokenize 

   

ps = PorterStemmer() 

  

# choose some words to be stemmed 

words = raw_train['text'] 

  

#for w in words: 

#    print(w, " : ", ps.stem(w))

#lemmatizing text column

# import these modules 

from nltk.stem import WordNetLemmatizer 

  

lemmatizer = WordNetLemmatizer()

  

# choose some words to be stemmed 

words = raw_train['text'] 

  

#for w in words: 

#    print(w, " : ", lemmatizer.lemmatize(w))