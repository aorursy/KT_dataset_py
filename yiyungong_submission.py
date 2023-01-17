import numpy as np 

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.svm import SVC



from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble
train_data = pd.read_csv("../input/clickbait-thumbnail-detection/train.csv", usecols=["class", "description", "viewCount", "likeCount","dislikeCount","commentCount","title"]) 

test_data = pd.read_csv("../input/clickbait-thumbnail-detection/test_2.csv", usecols=["ID", "description", "viewCount", "likeCount","dislikeCount","commentCount","title"])
#Text Preprocessing

##Remove Punctuation

import string

def remove_punctuation(text):

    punt = "".join([i for i in text if i not in string.punctuation])

    return punt
train_data['description'] = train_data['description'].apply(lambda x: remove_punctuation(x))

test_data['description'] = test_data['description'].apply(lambda x: remove_punctuation(x))
import re

#remove all the special characters

train_data['description'] = train_data['description'].apply(lambda x: re.sub(r'\W', ' ', x.lower()))

test_data['description'] = test_data['description'].apply(lambda x: re.sub(r'\W', ' ', x.lower()))

#remove all single character

train_data['description'] = train_data['description'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))

test_data['description'] = test_data['description'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))

#remove all single characters from the start

train_data['description'] = train_data['description'].apply(lambda x: re.sub(r'\^[a-zA-Z]\s+', ' ', x))

test_data['description'] = test_data['description'].apply(lambda x: re.sub(r'\^[a-zA-Z]\s+', ' ', x))

#substitute multiple spaces with a single space

train_data['description'] = train_data['description'].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))

test_data['description'] = test_data['description'].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))
#Sepreate Data

Y_train = train_data["class"]

X_train = train_data['description']

X_test = test_data['description']
#Encoding

from sklearn.preprocessing import LabelEncoder

Encoder = LabelEncoder()

Y_train = Encoder.fit_transform(Y_train)

#Y_train
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df= 3, sublinear_tf=True, norm='l2', ngram_range=(1, 2))

final_features = vectorizer.fit_transform(X_train).toarray()

#vectorizer.vocabulary_
train_X_Tfidf = vectorizer.transform(X_train)

test_X_Tfidf = vectorizer.transform(X_test)

#print(test_X_Tfidf)
#NB classifier

from sklearn import naive_bayes

classifier = naive_bayes.MultinomialNB()

classifier.fit(train_X_Tfidf, Y_train)

predictions_NB = classifier.predict(test_X_Tfidf)
#Applying K-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=train_X_Tfidf, y=Y_train, cv=10)

accuracies
#F1 Score from the training data

from sklearn.metrics import f1_score

predictions_train = classifier.predict(train_X_Tfidf)

f1_score(Y_train, predictions_train)
test_data["class"] = predictions_NB

test_data["class"] = test_data["class"].map(lambda x: "True" if x==1 else "False")

result = test_data[["ID","class"]]

result.head()
#render out result

result.to_csv("submission.csv", index=False)