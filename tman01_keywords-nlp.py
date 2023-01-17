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

authors = pd.read_csv("../input/nips-2015-papers/Authors.csv")

paper_authors = pd.read_csv("../input/nips-2015-papers/PaperAuthors.csv")

papers = pd.read_csv("../input/nips-2015-papers/Papers.csv")
authors.head(5)
paper_authors.head(5)
papers.describe()
papers.head(5)
papers.isnull().describe()
papers.drop(['EventType'],axis=1,inplace=True)

papers
papers['word_count']=papers['Abstract'].apply(lambda x:len(str(x).split(" ")))

papers[['Abstract','word_count']].head()
papers.word_count.describe()
freq = pd.Series(' '.join(papers['Abstract']).split()).value_counts()[:20]

freq
non_freq = pd.Series(' '.join(papers['Abstract']).split()).value_counts()[-20:]

non_freq
from nltk.stem.porter import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()

stem = PorterStemmer()

word = "good"

print("stemming:",stem.stem(word))

print("lemmatization:", lem.lemmatize(word, "v"))
import re

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer
##Creating a list of stopwords and adding a custom list of stopwords

stop_words=set(stopwords.words('english'))

print(stop_words)
##Creating a list of custom stopwords

new_words= ["using", "show", "result", "large", "also",

            "iv", "one", "two", "new", "previously", "shown"]

stop_words=stop_words.union(new_words)
corpus=[]

for i in range(0,403):

    #remove punctuations

    text=re.sub('^[a-zA-Z]',' ',papers.Abstract[i])

    #convert to lower case

    text=text.lower()

    #remove tags

    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    #remove special characters and digits

    text=re.sub('(\\d|\\W)',' ',text)

    #Convert to list from string

    list=text.split()

    #Stemming

    ps=PorterStemmer()

    #Lemmatization

    lem=WordNetLemmatizer()

    temp=[lem.lemmatize(word) for word in list if word not in stop_words]

    text=" ".join(temp)

    corpus.append(text)

    
from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

%matplotlib inline

wordcloud = WordCloud(background_color='white',stopwords=stop_words,

                                        max_words=100,

                                        max_font_size=50, 

                                        random_state=42 #near to std_dev

                                        ).generate(str(corpus))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

fig.savefig("word1.png", dpi=900)
from sklearn.feature_extraction.text import CountVectorizer

import re

cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))

X=cv.fit_transform(corpus)

#Most frequently occuring words

def get_top_n_words(corpus, n=None):

    vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in      

                   vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                       reverse=True)

    return words_freq[:n]#Convert most freq words to dataframe for plotting bar plot

top_words = get_top_n_words(corpus, n=20)

top_df = pd.DataFrame(top_words)

top_df.columns=["Word", "Freq"]#Barplot of most freq words

import seaborn as sns

sns.set(rc={'figure.figsize':(13,8)})

g = sns.barplot(x="Word", y="Freq", data=top_df)

g.set_xticklabels(g.get_xticklabels(), rotation=30)
#Most frequently occuring Bi-grams

def get_top_n2_words(corpus, n=None):

    vec1 = CountVectorizer(ngram_range=(2,2),  

            max_features=2000).fit(corpus)

    bag_of_words = vec1.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     

                  vec1.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                reverse=True)

    return words_freq[:n]

top2_words = get_top_n2_words(corpus, n=20)

top2_df = pd.DataFrame(top2_words)

top2_df.columns=["Bi-gram", "Freq"]

print(top2_df)#Barplot of most freq Bi-grams

import seaborn as sns

sns.set(rc={'figure.figsize':(13,8)})

h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)

h.set_xticklabels(h.get_xticklabels(), rotation=45)
#Most frequently occuring Tri-grams

def get_top_n3_words(corpus, n=None):

    vec1 = CountVectorizer(ngram_range=(3,3), 

           max_features=2000).fit(corpus)

    bag_of_words = vec1.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     

                  vec1.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                reverse=True)

    return words_freq[:n]

top3_words = get_top_n3_words(corpus, n=20)

top3_df = pd.DataFrame(top3_words)

top3_df.columns=["Tri-gram", "Freq"]

print(top3_df)#Barplot of most freq Tri-grams

import seaborn as sns

sns.set(rc={'figure.figsize':(13,8)})

j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)

j.set_xticklabels(j.get_xticklabels(), rotation=45)
from sklearn.feature_extraction.text import TfidfTransformer

 

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

tfidf_transformer.fit(X)# get feature names

feature_names=cv.get_feature_names()

 

# fetch document for which keywords needs to be extracted

doc=corpus[123]

 

#generate tf-idf for the given document

tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
#Function for sorting tf_idf in descending order

from scipy.sparse import coo_matrix

def sort_coo(coo_matrix):

    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

 

def extract_topn_from_vector(feature_names, sorted_items, topn=10):

    """get the feature names and tf-idf score of top n items"""

    

    #use only topn items from vector

    sorted_items = sorted_items[:topn]

 

    score_vals = []

    feature_vals = []

    

    # word index and corresponding tf-idf score

    for idx, score in sorted_items:

        

        #keep track of feature name and its corresponding score

        score_vals.append(round(score, 3))

        feature_vals.append(feature_names[idx])

 

    #create a tuples of feature,score

    #results = zip(feature_vals,score_vals)

    results= {}

    for idx in range(len(feature_vals)):

        results[feature_vals[idx]]=score_vals[idx]

    

    return results#sort the tf-idf vectors by descending order of scores

sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract only the top n; n here is 10

keywords=extract_topn_from_vector(feature_names,sorted_items,5)

 

# now print the results

print("\nAbstract:")

print(doc)

print("\nKeywords:")

for k in keywords:

    print(k,keywords[k])