# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/text-data-processing/train_twitter_data.csv', header=0)

train.head()
#Let's start with the basic features

#The Most basic feature is to get the "Number of words" in each tweet

#we just simply used split function in python

train['wordcount'] = train['tweet'].apply(lambda x: len(str(x).split(" ")))

train[['tweet','wordcount']].head()



#The Most basic feature is to get the "Number of characters" in each tweet

#we just simply get the string length function in python

train['totalcharcount'] = train['tweet'].str.len() 

train[['tweet','totalcharcount']].head()
#The Most basic feature is to get the "Average world length" in each tweet

#we just simply get the average words per tweet in python

def avg_word_calculation(sentence):

  words = sentence.split()

  return (sum(len(word) for word in words)/len(words))



train['averageword'] = train['tweet'].apply(lambda x: avg_word_calculation(x))

train[['tweet','averageword']].head()
#The Most basic feature is to get the "Number of stopwords" in each tweet

#we have imported stopwords from NLTK, which is a basic NLP library in python

from nltk.corpus import stopwords

stop = stopwords.words('english')



train['stopwords'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))

train[['tweet','stopwords']].head()
#The Most basic feature is to get the "Number of special character" in each tweet

#we just simply get the ‘starts with’ function because hashtags (or mentions) always appear at the beginning of a word.

train['hastags'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))

train[['tweet','hastags']].head()
#The Most basic feature is to get the "Number of numerics" in each tweet

#we just simply get the ‘isdigit’ function.

train['numerics'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

train[['tweet','numerics']].head()
#The Most basic feature is to get the "Number of Uppercase words" in each tweet

#we just simply get the ‘isupper’ function.

train['upper'] = train['tweet'].apply(lambda z: len([z for z in z.split() if z.isupper()]))

train[['tweet','upper']].head()
#The Most basic Preprocessing is to get the "Number of lower words" in each tweet

#we just compare the text - here Kaggle or kaggle both are different word

train['tweet'] = train['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))

train['tweet'].head()
#The Most basic Preprocessing is to get the "Removing Punctuation" in each tweet

#removing all instances of it will help us reduce the size of the training data.

train['tweet'] = train['tweet'].str.replace('[^\w\s]','')

train['tweet'].head()
#The Most basic Preprocessing is to get the "Removal of Stop Words" in each tweet

# stop words (or commonly occurring words) should be removed from the text data. create a list of stopwords ourselves or we can use predefined libraries.

from nltk.corpus import stopwords

stop = stopwords.words('english')

train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

train['tweet'].head()
#The Most basic Preprocessing is to get the "Common word removal" in each tweet

# we just removed commonly occurring words in a general sense.

freqt = pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]

print(freqt)
#The Most basic Preprocessing is to get the "Common word removal" in each tweet

# Now, let’s remove these words as their presence will not of any use in classification of our text data.



freqt = list(freqt.index)

train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freqt))

train['tweet'].head()

#The Most basic Preprocessing is to get the "Rare words removal" in each tweet

# just as we removed the most common words



freqt = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]

freqt
#The Most basic Preprocessing is to get the "Rare words removal" in each tweet

# just as we removed the most common words



freqt = list(freqt.index)

train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freqt))

train['tweet'].head()
#The Most basic Preprocessing is to get the "Spelling correction" in each tweet

# To achieve this we will use the textblob library. If you are not familiar with it, you can check my previous article on ‘NLP for beginners using textblob’.



from textblob import TextBlob

train['tweet'][:5].apply(lambda x: str(TextBlob(x).correct()))
#The Most basic Preprocessing is to get the "Tokenization" in each tweet

# To achieve this we will use the textblob library. If you are not familiar with it, you can check my previous article on ‘NLP for beginners using textblob’.



TextBlob(train['tweet'][1]).words

#WordList(['thanks', 'lyft', 'credit', 'cant', 'use', 'cause', 'dont', 'offer', 'wheelchair', 'vans', 'pdx', 'disapointed', 'getthanked'])
#The Most basic Preprocessing is to get the "Stemming" in each tweet

# Stemming refers to the removal of suffices, like “ing”, “ly”, “s”, etc. by a simple rule-based approach. For this purpose, we will use PorterStemmer from the NLTK library.



from nltk.stem import PorterStemmer

st = PorterStemmer()

train['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


#The Most basic Preprocessing is to get the "Lemmatization" in each tweet

# Lemmatization is a more effective option than stemming because it converts the word into its root word.



from textblob import Word

train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

train['tweet'].head()
#The Most basic Preprocessing is to get the "Term frequency" in each tweet

# F = (Number of times term T appears in the particular row) / (number of terms in that row)



TextBlob(train['tweet'][0]).ngrams(2)
#The Most basic Preprocessing is to get the "N-grams" in each tweet

# N-grams are the combination of multiple words used together. Ngrams with N=1 are called unigrams. Similarly, bigrams (N=2), trigrams (N=3) and so on can also be used



tf1 = (train['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()

tf1.columns = ['words','tf']

tf1
#The Most basic Preprocessing is to get the "Inverse Document Frequency" in each tweet

# IDF = log(N/n), where, N is the total number of rows and n is the number of rows in which the word was present.



for i,word in enumerate(tf1['words']):

  tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['tweet'].str.contains(word)])))



tf1
#The Most basic Preprocessing is to get the "Term Frequency – Inverse Document Frequency (TF-IDF)" in each tweet

# TF-IDF is the multiplication of the TF and IDF which we calculated above.



tf1['tfidf'] = tf1['tf'] * tf1['idf']

tf1
#The Most basic Preprocessing is to get the "Term Frequency – Inverse Document Frequency (TF-IDF)" in each tweet

# TF-IDF is the multiplication of the TF and IDF which we calculated above.



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,1))

train_vect = tfidf.fit_transform(train['tweet'])



from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")

train_bow = bow.fit_transform(train['tweet'])

#train_bow
#Sentiment Analysis

train['tweet'][:5].apply(lambda x: TextBlob(x).sentiment)
train['sentiment'] = train['tweet'].apply(lambda x: TextBlob(x).sentiment[0] )

train[['tweet','sentiment']].head()
#Word Embeddings

from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = '../input/glove6b100dtxt/glove.6B.100d.txt'

word2vec_output_file = 'glove.6B.100d.txt.word2vec'

glove2word2vec(glove_input_file, word2vec_output_file)
#Now, we can load the above word2vec file as a model.



from gensim.models import KeyedVectors # load the Stanford GloVe model

filename = 'glove.6B.100d.txt.word2vec'

model = KeyedVectors.load_word2vec_format(filename, binary=False)
#Let’s say our tweet contains a text saying ‘go away’. We can easily obtain it’s word vector using the above model:



model['go']
model['away']
#We then take the average to represent the string ‘go away’ in the form of vectors having 100 dimensions.



(model['go'] + model['away'])/2