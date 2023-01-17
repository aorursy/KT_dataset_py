# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



#load packages

import sys #access to system parameters https://docs.python.org/3/library/sys.html

print("Python version: {}". format(sys.version))



import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features

print("pandas version: {}". format(pd.__version__))



import matplotlib #collection of functions for scientific and publication-ready visualization

print("matplotlib version: {}". format(matplotlib.__version__))



import numpy as np #foundational package for scientific computing

print("NumPy version: {}". format(np.__version__))



import scipy as sp #collection of functions for scientific computing and advance mathematics

print("SciPy version: {}". format(sp.__version__)) 



import IPython

from IPython import display #pretty printing of dataframes in Jupyter notebook

print("IPython version: {}". format(IPython.__version__)) 



import sklearn #collection of machine learning algorithms

print("scikit-learn version: {}". format(sklearn.__version__))



#misc libraries

import random

import time





#ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('-'*25)



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns





#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
raw_data = pd.read_csv("/kaggle/input/us-consumer-finance-complaints/consumer_complaints.csv")

raw_data.head()
raw_data.sample(5,random_state=89)
##find missing values

raw_data.isnull().mean().round(4)*100
## We only want to analyze the data that has actual text complaints included, so i'm going to remove all data without that information in the column



raw_data.dropna(subset = ["consumer_complaint_narrative"], inplace=True)



##find missing values Now we can see we have no nulls in that specific column

raw_data.isnull().mean().round(4)*100





raw_data.head()
# Make the index Product, so that we can use it as the dependent variable



raw_data2 = raw_data[['product', 'consumer_complaint_narrative']].copy()





raw_data3 = raw_data2.set_index('product')



raw_data3



raw_data3.consumer_complaint_narrative['Mortgage']





# Apply a first round of text cleaning techniques

import re

import string



def clean_text_round1(text):

    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



round1 = lambda x: clean_text_round1(x)
# Let's take a look at the updated text

raw_data4 = pd.DataFrame(raw_data3.consumer_complaint_narrative.apply(round1))

raw_data4
# now everything is lowercase, and there is less punctuation

raw_data4.head()


# Apply a second round of cleaning

def clean_text_round2(text):

    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''

    text = re.sub('[‘’“”…]', '', text)

    text = re.sub('\n', '', text)

    return text



round2 = lambda x: clean_text_round2(x)


raw_data5 = pd.DataFrame(raw_data4.consumer_complaint_narrative.apply(round2))



raw_data5
import pickle



# Let's pickle it for later use

raw_data5.to_pickle("corpus.pkl")

##Since there are so many rows in this data, I want to only show the unique products, therefore I will join and concatenate all the consumer complaints into one cell essentially of data



raw_data6 = raw_data5.groupby(['product']).agg(lambda col: ','.join(col))



# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words

from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(stop_words='english')

data_cv = cv.fit_transform(raw_data6.consumer_complaint_narrative)

data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())

data_dtm.index = raw_data6.index
data_dtm
data_dtm.to_pickle("dtm1.pkl")
#Let's also pickle the cleaned data (before we put it in document-term matrix format) and the CountVectorizer object

raw_data5.to_pickle('data_clean.pkl')

pickle.dump(cv, open("cv.pkl", "wb"))
#Read in the document-term matrix

import pandas as pd 

import pickle



data = pd.read_pickle('dtm1.pkl')

data = data.transpose()

data.head()
# Find the top 30 words said by each product complaint

top_dict = {}

for c in data.columns:

    top = data[c].sort_values(ascending=False).head(30)

    top_dict[c]= list(zip(top.index, top.values))



top_dict





# Print the top 15 words said by each product complaint

for product, top_words in top_dict.items():

    print(product)

    print(', '.join([word for word, count in top_words[0:14]]))

    print('---')
# by looking at these top words, you can see that some of them have very little meaning and could be added to a stop words list, so let's do just that.

# Look at the most common top words --> add them to the stop word list



from collections import Counter



# Let's first pull out the top 30 words for each comedian

words = []

for product in data.columns:

    top = [word for (word, count) in top_dict[product]]

    for t in top:

        words.append(t)

        

words
# Let's aggregate this list and identify the most common words along with how many routines they occur in

Counter(words).most_common()


# If more than 6 of the products have it as a top word, exclude it from the list

add_stop_words = [word for word, count in Counter(words).most_common() if count > 6]

add_stop_words


# Let's make some word clouds!



from wordcloud import WordCloud



wc = WordCloud(background_color="white", colormap="Dark2",

               max_font_size=150, random_state=42)
# Reset the output dimensions

import matplotlib.pyplot as plt



plt.rcParams['figure.figsize'] = [16, 6]



full_names = raw_data6.index





# Create subplots for each comedian

for index, product in enumerate(data.columns):

    wc.generate(raw_data6.consumer_complaint_narrative[product])

    

    plt.subplot(3, 4, index+1)

    plt.imshow(wc, interpolation="bilinear")

    plt.axis("off")

    plt.title(full_names[index])

    

plt.show()
# Find the number of unique words that each product



# Identify the non-zero items in the document-term matrix, meaning that the word occurs at least once

unique_list = []

for product in data.columns:

    uniques = data[product].nonzero()[0].size

    unique_list.append(uniques)



# Create a new dataframe that contains this unique word count

data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=['product', 'unique_words'])

data_unique_sort = data_words.sort_values(by='unique_words')

data_unique_sort
total_list = []

for product in data.columns:

    totals = sum(data[product])

    total_list.append(totals)

    

print(total_list)





data_words['total_words'] = total_list
data_words
#Unique words per product

sns.catplot(x="product",y="unique_words",kind='bar',data=data_words, height = 10, aspect = 2.25)
# Total words per product

sns.catplot(x="product",y="total_words",kind='bar',data=data_words, height = 10, aspect = 2.25, legend = True, legend_out = True)


# We'll start by reading in the corpus, which preserves word order

import pandas as pd



data = pd.read_pickle('corpus.pkl')

data
raw_data6.head()


# Create quick lambda functions to find the polarity and subjectivity of each routine

# Terminal / Anaconda Navigator: conda install -c conda-forge textblob

from textblob import TextBlob



pol = lambda x: TextBlob(x).sentiment.polarity

sub = lambda x: TextBlob(x).sentiment.subjectivity



raw_data6['polarity'] = raw_data6['consumer_complaint_narrative'].apply(pol)

raw_data6['subjectivity'] = raw_data6['consumer_complaint_narrative'].apply(sub)

raw_data6
full_names
sns.scatterplot(x='polarity', y='subjectivity', hue = full_names, data=data)
# placeholder for sentiment analysis over time
#Topic Modeling



data_dtm
#Import the necessary modules for LDA with gensim

# Terminal / Anaconda Navigator: conda install -c conda-forge gensim

from gensim import matutils, models

import scipy.sparse



# import logging

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# One of the required inputs is a term-document matrix

tdm = data_dtm.transpose()

tdm.head()
#We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus

sparse_counts = scipy.sparse.csr_matrix(tdm)

corpus = matutils.Sparse2Corpus(sparse_counts)
# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix

cv = pickle.load(open("cv.pkl", "rb"))

id2word = dict((v, k) for k, v in cv.vocabulary_.items())
# Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),

# we need to specify two other parameters as well - the number of topics and the number of passes

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=2, passes=10)

lda.print_topics()


# LDA for num_topics = 3

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=3, passes=10)

lda.print_topics()


# LDA for num_topics = 4

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, passes=10)

lda.print_topics()
# Let's create a function to pull out nouns from a string of text

from nltk import word_tokenize, pos_tag



def nouns(text):

    '''Given a string of text, tokenize the text and pull out only the nouns.'''

    is_noun = lambda pos: pos[:2] == 'NN'

    tokenized = word_tokenize(text)

    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 

    return ' '.join(all_nouns)
raw_data6
# Let's create a function to pull out nouns from a string of text

def nouns_adj(text):

    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''

    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'

    tokenized = word_tokenize(text)

    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 

    return ' '.join(nouns_adj)
# Create a new document-term matrix using only nouns and adjectives, also remove common words with max_df

cvna = CountVectorizer(stop_words=stop_words, max_df=.8)

data_cvna = cvna.fit_transform(data_nouns_adj.transcript)

data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())

data_dtmna.index = data_nouns_adj.index

data_dtmna
data_dtm