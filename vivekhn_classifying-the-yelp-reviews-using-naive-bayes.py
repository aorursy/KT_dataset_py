# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# load dataset

yelp_df = pd.read_csv("/kaggle/input/yelp-reviews/yelp.csv")
# examine the data

yelp_df.head()
# print the summary statistics of numercial columns

yelp_df.describe()
# Verify the datatypes and check for any nulls

yelp_df.info()
# print the example text review

yelp_df.text[0]
# calculate the length of reviews

yelp_df['length'] = yelp_df.text.apply(len)
# plot the histogram for the length values

yelp_df.length.plot(bins = 20, kind='hist')
yelp_df.length.describe()
# print the review with highest chars 

yelp_df[yelp_df.length==4997].text.iloc[0]
# print the reviews with lowest char

yelp_df[yelp_df.length == 1].text.iloc[0]
# plot the count of reviews

sns.countplot(yelp_df.stars,palette='GnBu_d')
# plot histograms for each stars

g =sns.FacetGrid(data =yelp_df,col='stars',col_wrap=3)

g.map(plt.hist,'length',bins=20,color='orange')
# prepare the data for prediction

sns.countplot(yelp_df[yelp_df.stars!=3].stars)
# exclude all records having with star 3

yelp_df = yelp_df[yelp_df.stars!=3]
# create function to calcualte the target value

def create_target(stars):

    if stars<3:

        target = 0#

    else:

        target = 1

    return target
# store the value into target column

yelp_df['target'] = yelp_df.stars.apply(create_target)
# examine the values

yelp_df[['stars','target']]
# remove punctuation

import string

string.punctuation
# remove stopwords

from nltk.corpus import stopwords

stopwords.words('english')
# defining the fuction to remove punctuations & stop words

def text_cleaning(text):

    remove_punctuation = ''.join([char for char in text if char not in string.punctuation])

    remove_stopwords = [word for word in remove_punctuation.split() if word.lower() not in stopwords.words('english')]

    return remove_stopwords
# count vectorization ( 2d matrix containing word frequency)

from sklearn.feature_extraction.text import CountVectorizer

CountVectorizer = CountVectorizer(analyzer = text_cleaning)

yelp_vectorizer=CountVectorizer.fit_transform(yelp_df.text)
yelp_vectorizer.shape
X =yelp_vectorizer

y = yelp_df.target.values.reshape(-1,1)

print(X.shape)

print(y.shape)
# split the data into train and test

from sklearn.model_selection import train_test_split

X_train ,X_test, y_train, y_test = train_test_split(X,y,test_size =0.2)
# train model

from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()

NB_classifier.fit(X_train,y_train)
# print the confusion matrix for trained data

from sklearn.metrics import classification_report,confusion_matrix

predict_train = NB_classifier.predict(X_train)

cm = confusion_matrix(y_train,predict_train)

sns.heatmap(cm,annot =True,cmap="Blues")

plt.ylabel("Actual")

plt.xlabel("Predicted")

print(cm)

print(classification_report(y_train,predict_train))
# print the confusion matrix for test data

predict_test = NB_classifier.predict(X_test)

cm = confusion_matrix(y_test,predict_test)

sns.heatmap(cm,annot =True,cmap="Blues")

plt.ylabel("Actual")

plt.xlabel("Predicted")

print(cm)

print(classification_report(y_test,predict_test))