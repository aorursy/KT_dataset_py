import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re

from time import time

import string

#import itertools

from pprint import pprint

from nltk import PorterStemmer

from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report
df = pd.read_csv("../input/textdb3/fake_or_real_news.csv")

print(df.shape)
df.head()
df.describe()
df.info ()
df.isnull().any()
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import math as math

from pylab import rcParams



%matplotlib inline
plt.figure (figsize=(6,6))

p = sns.countplot(data=df,x = 'label',)
df.loc[df['label']== 0, 'label'] = 'REAL'

df.loc[df['label']== 1, 'label'] = 'FAKE'

df.columns

df['label'].value_counts()
# Draw a graph of text length verse frequency



import matplotlib

%matplotlib inline

df['text'].str.len().plot(kind = 'hist', bins = 1000, figsize = (12,5))
y = df.label 

df.drop("label", axis=1) 

X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=42)
print(X_train.shape)

print(type(X_train))

print(X_train.head())

print(X_test.shape)

print(type(X_test))

print(X_test.head())
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) 

tfidf_train = tfidf_vectorizer.fit_transform(X_train) 

tfidf_test = tfidf_vectorizer.transform(X_test)
print(tfidf_train.shape)

print(tfidf_test.shape)
print(tfidf_vectorizer.get_feature_names()[-10:])
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
count_vectorizer = CountVectorizer(stop_words='english')

count_train = count_vectorizer.fit_transform(X_train)

count_test = count_vectorizer.transform(X_test)
count_vectorizer.get_feature_names()[:10]
tfidf_df.head()
import sklearn.metrics as metrics
def classify_and_fit(clf, X_train, y_train, X_test, y_test, class_labels = ['FAKE', 'REAL']):

    print("Classifier : ", clf )

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    score = metrics.accuracy_score(y_test, pred)

    print("Accuracy:   %0.3f" % score)

    print("\nConfusion Matrix :")

    #print(pd.crosstab(y_test, pred, rownames=['True'], colnames=['Predicted'], margins=True))

    cm = metrics.confusion_matrix(y_test, pred, labels=class_labels)

    print(cm)

    print("\nReport :")    

    print(classification_report(y_test, pred, target_names=class_labels))

    return clf
clf = MultinomialNB() 

classify_and_fit(clf, tfidf_train, y_train, tfidf_test, y_test)
linear_clf = PassiveAggressiveClassifier()

classify_and_fit(linear_clf, tfidf_train, y_train, tfidf_test, y_test)
log_reg = LogisticRegression()

classify_and_fit(log_reg, tfidf_train, y_train, tfidf_test, y_test)
from sklearn.ensemble import RandomForestClassifier

ran_class= RandomForestClassifier()

classify_and_fit(ran_class, tfidf_train, y_train, tfidf_test, y_test)
def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):

    class_labels = classifier.classes_

    feature_names = vectorizer.get_feature_names()

    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]

    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:

        print(class_labels[0], coef, feat)

    print()

    for coef, feat in reversed(topn_class2):

        print(class_labels[1], coef, feat)





most_informative_feature_for_binary_classification(tfidf_vectorizer, linear_clf, n=30)
from wordcloud import WordCloud 

# Start with one review:

text = df.text[0]



# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()