# Run this cell and select the kaggle.json file downloaded

# from the Kaggle account settings page.



#from google.colab import files

#files.upload()  #Kaggle>account>API json file
# Let's make sure the kaggle.json file is present.

#!ls -lha kaggle.json
# Next, install the Kaggle API client.

#!pip install -q kaggle
# The Kaggle API client expects this file to be in ~/.kaggle,

# so move it there.



#!mkdir -p ~/.kaggle

#!cp kaggle.json ~/.kaggle/



# This permissions change avoids a warning on Kaggle tool startup.



#!chmod 600 ~/.kaggle/kaggle.json
# Copy the kaggle data set locally.

#!kaggle datasets download -d datafiniti/consumer-reviews-of-amazon-products
#! unzip -q -n 'consumer-reviews-of-amazon-products.zip'
#!ls
#Importing necessary libraries

import pandas as pd

from nltk.corpus import stopwords

import string

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
#Creating dataframe of amazon reviews from csv

df=pd.read_csv('../input/consumer-reviews-of-amazon-products/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')

df.columns
#Shape of dataframe

df.shape
#Filtering Columns

df=df[['reviews.rating' , 'reviews.text' , 'reviews.title']]
df.head()
df.info()
#Checking for null values

print(df.isnull().sum())
#ploting graph on the basis of review ratings

df["reviews.rating"].value_counts().sort_values().plot.bar()
#Rating value counts

df['reviews.rating'].value_counts()
sentiment = {1: 0,

            2: 0,

            3: 1,

            4: 2,

            5: 2}



df["sentiment"] = df["reviews.rating"].map(sentiment)
df.head()
#sentiment value counts

df.sentiment.value_counts()
#implementing bag of words using CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import RegexpTokenizer

#tokenizer to remove unwanted elements from out data like symbols and numbers

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)

text_counts= cv.fit_transform(df['reviews.text'])
print(text_counts[0])
#Split train and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(text_counts, df['reviews.rating'], test_size=0.3, random_state=1)
from sklearn.naive_bayes import MultinomialNB

# Model Generation Using Multinomial Naive Bayes

clf = MultinomialNB().fit(X_train, y_train)

predicted= clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

print("BoW MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
#Using TF-IDF approach

from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer()

text_tf= tf.fit_transform(df['reviews.text'])
#Split train and test set

X_train, X_test, y_train, y_test = train_test_split(text_tf, df['reviews.rating'], test_size=0.3, random_state=1)
#Again Generating model Using Multinomial Naive Bayes

clf = MultinomialNB().fit(X_train, y_train)

predicted= clf.predict(X_test)

print("TF-IDF MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
# Random Forests

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier().fit(X_train, y_train)

predicted= classifier.predict(X_test)

print("Random Forests Accuracy:",metrics.accuracy_score(y_test, predicted))