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
#Read the csv file

df = pd.read_csv("/kaggle/input/nips-papers/papers.csv")



#select only that has abstract

df = df[df['abstract'] != 'Abstract Missing']

df.head()
#fix the index

df.index = range(0,len(df.index))
#Combine title and abstract columns and keep only id, year & abstract columns

df['abstract1'] = df['title'] + df['abstract']

df = df.drop(['title', 'event_type', 'pdf_name','abstract','paper_text'], axis=1)

df.head()
#Add word count

df['word_count'] = df['abstract1'].apply(lambda x : len(str(x).split(" ")))

df.head()
#Just to get idea of volume of data

print(len(df.index))

print(df['word_count'].describe())
#Just to get idea of frequency of wordss

most_common_words = pd.Series(''.join(df['abstract1']).split()).value_counts()[:20]

least_common_words = pd.Series(''.join(df['abstract1']).split()).value_counts()[-20:]

print(most_common_words)

print(least_common_words)
#construct stop words

from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

new_stop_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]

stop_words = stop_words.union(new_stop_words)
#construct the corpus of unique words that matter

#remove the stop words

from nltk.stem.wordnet import WordNetLemmatizer

import re



corpus = []

for i in range(0,len(df.index)):

    #remove puncuations

    text = re.sub('[^a-zA-Z]', ' ', df['abstract1'][i])

    

    #convert to lower case

    text = text.lower()

    

    #remove tags

    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    

    # remove special characters and digits

    text=re.sub("(\\d|\\W)+"," ",text)

    

    #split into words

    text = text.split()

    

    #Lemmatize

    lem = WordNetLemmatizer()

    text = [lem.lemmatize(word) for word in text if word not in stop_words]

    text = ' '.join(text)

    corpus.append(text)

#view an item from corpus

corpus[24]
#see the most common words now

#corpus_series = pd.Series(corpus)

most_common_in_corpus = pd.Series(''.join(corpus).split()).value_counts()[:20]

#corpus_series.head()

most_common_in_corpus
# build vocabulary of 10000 words and fit the corpus

# uses bag or words concept

from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(stop_words = stop_words, ngram_range= (1,3), max_df = 0.8, max_features = 10000)

X = cv.fit_transform(corpus)
# to understand how max_df parametr works in CountVectorizer

sample = [

     'This is the first document.',

     'This document is the second document.',

     'And this is the third one.',

     'Is this the first document?'

 ]





#vectorizer = CountVectorizer()

# max_df parameter removes noise from corpus. range 0 - 1.0

# when max_df is ignored, the following unique words are considered

# vectorizer.get_feature_names() give ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']



# if max_df = 0.8, the following unique words are considerd

# note that 'is' (freq = 4) ; the' (freq = 4) ; this (freq = 4) are all ignored since their occurance is more than 80%

# vectorizer.get_feature_names() gives ['and', 'document', 'first', 'one', 'second', 'third']

vectorizer = CountVectorizer(max_df=0.8)

V = vectorizer.fit_transform(sample)

vectorizer.get_feature_names()
# to understand how to plot the frequency bar chart

vec = CountVectorizer().fit(sample)

bag_of_words = vec.transform(sample)



# above 2 lines are simlar to the following 2 lines

# vec = CountVectorizer()

# bag_of_words = vec.fit_transform(sample)



print(vec.get_feature_names())

print(bag_of_words.sum(axis=0)) #Gives a 2d numpy matrix

print(sorted(vec.vocabulary_.items(), key= lambda x: x[1])) #gives the word and index

vec.vocabulary_.items()
#To visualize data. Not used in algorithm

#See the data how it looks when vectorization is done with 1-gram

#Most frequently occuring 1-grams

import matplotlib.pyplot as plt

def get_top_n_words(corpus, n=None):

    vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) #Refer above how .sum(axis=0) works

    

    #constructs the tuple with word and its frequency of occurancy

    # like (and, 1), (document, 4), (first, 2) for the sample example

    # the first element of the tuple is the word that comes from vocabulary_items()

    # the second element is the freq of word from sum_words

    words_freq = [(word, sum_words[0, index]) for word, index in vec.vocabulary_.items()]  

    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return words_freq[:n]



#convert the word frequency in dataframe

top_1gram_df = pd.DataFrame(get_top_n_words(corpus,n=20), columns=['Word','Freq'])

#print(top_1gram_df.head())



#plot in a bar chart

plt.bar(top_1gram_df['Word'],top_1gram_df['Freq'])

plt.xticks(rotation=70)

plt.xlabel('Word')

plt.ylabel('Frequency')

plt.title('Word Frequency')

plt.show()
#To visualize data. Not used in algorithm

#See the data how it looks when vectorization is done with 2-grams

#Most frequently occuring 2-grams

def get_top_n2_words(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2,2), max_features=2000).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0)

    words_freq = [(word,sum_words[0,index]) for word,index in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key= lambda x: x[1], reverse=True)

    return words_freq[:n]



#Create Dataframe to create bar chart

top_2gram_df = pd.DataFrame(get_top_n2_words(corpus,n=20),columns=['2gram Words','Freq'])



#Create bar chart

plt.bar(top_2gram_df['2gram Words'],top_2gram_df['Freq'])

plt.xticks(rotation = 80)

plt.xlabel('2gram Word')

plt.ylabel('Frequency')

plt.title('2gram Frequency')

plt.show()
#To visualize data. Not used in algorithm

#See the data how it looks when vectorization is done with 2-grams

#Most frequently occuring 2-grams

def get_top_n3_words(corpus, n=None):

    vec = CountVectorizer(ngram_range=(3,3), max_features=2000).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0)

    words_freq = [(word, sum_words[0,index]) for word, index in vec.vocabulary_.items()]

    words_freq = sorted(words_freq, key= lambda x: x[1], reverse=True)

    return words_freq[:n]



#Create Dataframe to blot bar chart

top_3gram_df = pd.DataFrame(get_top_n3_words(corpus,n=20),columns=['3gram words','Freq'])



#plot bar chart

plt.bar(top_3gram_df['3gram words'], top_3gram_df['Freq'])

plt.xticks(rotation=80)

plt.xlabel('3gram Words')

plt.ylabel('Frequency')

plt.title('3gram words frequency')

plt.show()
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

tfidf_transformer.fit(X)



doc = corpus[500]



tfidf_vector = tfidf_transformer.transform(cv.transform([doc]))
feature_names = cv.get_feature_names()

def extract_and_sort(coo_matrix):

    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key= lambda x : (x[1], x[0]),reverse=True)



def get_topn_features_from_vector(sorted_items, feature_names, topn=10):

    sorted_items = sorted_items[:topn]

    

    feature_vals = []

    score_vals = []

    

    for index, score in sorted_items:

        feature_vals.append(feature_names[index])

        score_vals.append(score)

    

    result = {}

    for i in range(len(feature_vals)):

        result[feature_vals[i]] = score_vals[i]

    

    return result

    

    

sorted_items = extract_and_sort(tfidf_vector.tocoo())

keywords = get_topn_features_from_vector(sorted_items, feature_names, topn=5)



print('Document')

print(doc)

print()

print('Keywords')

for k in keywords:

    print(k , ': ', keywords[k])