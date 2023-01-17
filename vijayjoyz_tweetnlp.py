import numpy as np 

import pandas as pd 



# text processing libraries

import re

import string

import nltk

from nltk.corpus import stopwords



# XGBoost

import xgboost as xgb

from xgboost import XGBClassifier



# sklearn 

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns



# File system manangement

import os
root_path = "../input/nlp-getting-started"

train_path = os.path.join(root_path,"train.csv")

train = pd.read_csv(train_path)

train.head()
test_path = os.path.join(root_path,"test.csv")

test = pd.read_csv(test_path)

test.head()
train['target'].value_counts()
sns.barplot(x=(train['target']==1).value_counts(),y=train['target'].value_counts(),palette="magma",data=train)
train['text'][:10]
# A disaster tweet

disaster_tweets = train[train['target']==1]['text']

disaster_tweets.values[1]
#not a disaster tweet

non_disaster_tweets = train[train['target']==0]['text']

non_disaster_tweets.values[1]
from wordcloud import WordCloud

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20, 8])

wordcloud1 = WordCloud( background_color='black',

                        width=600,

                        height=400).generate(" ".join(disaster_tweets))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('Disaster Tweets',fontsize=40);



wordcloud2 = WordCloud( background_color='black',

                        width=600,

                        height=400).generate(" ".join(non_disaster_tweets))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Non Disaster Tweets',fontsize=40);
def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



# Applying the cleaning function to both test and training datasets

train['text'] = train['text'].apply(lambda x: clean_text(x))

test['text'] = test['text'].apply(lambda x: clean_text(x))



# Let's take a look at the updated text

train['text'].head()
text = "Are you coming , aren't you"

tokenizer1 = nltk.tokenize.WhitespaceTokenizer()

tokenizer2 = nltk.tokenize.TreebankWordTokenizer()

tokenizer3 = nltk.tokenize.WordPunctTokenizer()

tokenizer4 = nltk.tokenize.RegexpTokenizer(r'\w+')



print("Example Text: ",text)

print("------------------------------------------------------------------------------------------------")

print("Tokenization by whitespace:- ",tokenizer1.tokenize(text))

print("Tokenization by words using Treebank Word Tokenizer:- ",tokenizer2.tokenize(text))

print("Tokenization by punctuation:- ",tokenizer3.tokenize(text))

print("Tokenization by regular expression:- ",tokenizer4.tokenize(text))
# Tokenizing the training and the test set

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))

test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))

train['text'].head()
#Needed While you run in COLAB

nltk.download('stopwords')
def remove_stopwords(text):

    """

    Removing stopwords belonging to english language

    

    """

    words = [w for w in text if w not in stopwords.words('english')]

    return words





train['text'] = train['text'].apply(lambda x : remove_stopwords(x))

test['text'] = test['text'].apply(lambda x : remove_stopwords(x))

train.head()
#Needed While you run in COLAB

nltk.download('wordnet')
# Stemming and Lemmatization examples

text = "feet cats wolves talked"



tokenizer = nltk.tokenize.TreebankWordTokenizer()

tokens = tokenizer.tokenize(text)



# Stemmer

stemmer = nltk.stem.PorterStemmer()

print("Stemming the sentence: ", " ".join(stemmer.stem(token) for token in tokens))



# Lemmatizer

lemmatizer=nltk.stem.WordNetLemmatizer()

print("Lemmatizing the sentence: ", " ".join(lemmatizer.lemmatize(token) for token in tokens))
def combine_text(list_of_text):

    '''Takes a list of text and combines them into one large chunk of text.'''

    combined_text = ' '.join(list_of_text)

    return combined_text



train['text'] = train['text'].apply(lambda x : combine_text(x))

test['text'] = test['text'].apply(lambda x : combine_text(x))

train['text']

train.head()
def text_preprocessing(text):

    """

    Cleaning and parsing the text.



    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(remove_stopwords)

    return combined_text
count_vectorizer = CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train['text'])

test_vectors = count_vectorizer.transform(test["text"])



## Keeping only non-zero elements to preserve space 

print(train_vectors[0].todense())
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_tfidf = tfidf.fit_transform(train['text'])

test_tfidf = tfidf.transform(test["text"])
clf_NB_TFIDF = MultinomialNB()

scores = model_selection.cross_val_score(clf_NB_TFIDF, train_tfidf, train["target"], cv=5, scoring="f1")

scores
clf_NB_TFIDF.fit(train_tfidf, train["target"])
def submission(submission_file_path,model,test_vectors):

    sample_submission = pd.read_csv(submission_file_path)

    sample_submission["target"] = model.predict(test_vectors)

    sample_submission.to_csv("submission.csv", index=False)
submission_file_path = os.path.join(root_path,"sample_submission.csv")

test_vectors=test_tfidf

submission(submission_file_path,clf_NB_TFIDF,test_vectors)