import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

#nltk.download('wordnet')
data = pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv")
sns.countplot(data["Recommended IND"])

plt.show()
data = data.drop((data[data["Review Text"].isnull()]).index)

label1 = data[data["Recommended IND"] == 1].iloc[:5000] # may not be good method i don't know for sure 

label0 = data[data["Recommended IND"] == 0]

data = pd.concat([label0,label1], axis=0)

data = data[["Recommended IND","Review Text" ]]
sns.countplot(data["Recommended IND"])

plt.show()
import re

def clean_and_tokenize(review):

    text = review.lower()

    

    tokenizer = nltk.tokenize.TreebankWordTokenizer()

    tokens = tokenizer.tokenize(text)

    

    stemmer = nltk.stem.WordNetLemmatizer()

    text = " ".join(stemmer.lemmatize(token) for token in tokens)

    text = re.sub("[^a-z']"," ", text)

    return text

data["Review Text"] = data["Review Text"].apply(clean_and_tokenize)
x_data = data["Review Text"]

y= data["Recommended IND"]
from sklearn.feature_extraction.text import CountVectorizer



count_vectorizer = CountVectorizer(max_features=5000,ngram_range=(1, 2), stop_words = "english")



features = count_vectorizer.fit_transform(x_data)

count_vec_x = pd.DataFrame(

    features.todense(),

    columns=count_vectorizer.get_feature_names()

)
from sklearn.model_selection import train_test_split

x_train_cv, x_test_cv, y_train_cv, y_test_cv = train_test_split(count_vec_x,y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train_cv,y_train_cv)

print("Score:", nb.score(x_test_cv,y_test_cv))
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(x_train_cv,y_train_cv)

print("Score: ", lr.score(x_test_cv,y_test_cv))
from sklearn.feature_extraction.text import TfidfVectorizer

# using default tokenizer in TfidfVectorizer

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2),max_features=5000,stop_words = "english")

features2 = tfidf.fit_transform(x_data)

tfidf_x = pd.DataFrame(

    features2.todense(),

    columns=tfidf.get_feature_names()

)
from sklearn.model_selection import train_test_split

x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_x,y, test_size=0.2, random_state=42)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train_tfidf,y_train_tfidf)

print("Score:", nb.score(x_test_tfidf,y_test_tfidf))
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(x_train_tfidf,y_train_tfidf)

print("Score: ", lr.score(x_test_tfidf,y_test_tfidf))