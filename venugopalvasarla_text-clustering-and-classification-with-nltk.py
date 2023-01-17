import re

import nltk

from nltk.corpus import reuters

import numpy as np

import pandas as pd

from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.preprocessing.label import MultiLabelBinarizer    

from sklearn.feature_extraction.text import TfidfVectorizer

import warnings

warnings.filterwarnings('ignore')
stopWords = stopwords.words('english')

charfilter = re.compile('[a-zA-Z]+')
#let's create a tokenizer:

def simple_tokenizer(text):

    #tokenizing the words:

    words = word_tokenize(text)

    #converting all the tokens to lower case:

    words = map(lambda word: word.lower(), words)

    #let's remove every stopwords

    words = [word for word in words if word not in stopWords]

    #stemming all the tokens

    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))

    ntokens = list(filter(lambda token : charfilter.match(token),tokens))

    return ntokens
#let's vectorize the data using tfidf vector:

vec = TfidfVectorizer(tokenizer = simple_tokenizer, max_features = 1000, norm = 'l2')
#let's create the dataset

test_data = []

test_labels = []

train_data = []

train_labels = []

for file in reuters.fileids():

    if file.startswith('training/'):

        train_data.append(reuters.raw(file))

        train_labels.append(reuters.categories(file))

    elif file.startswith('test/'):

        test_data.append(reuters.raw(file))

        test_labels.append(reuters.categories(file))

    else:

        print('error')
#let's consider only 5 categories

our_labels = ['crude', 'sugar', 'coffee', 'ship', 'gold']
docs_pca_train = []

docs_pca_labels = []

for i in range(len(train_labels)):

    if any(item in train_labels[i] for item in our_labels):

        docs_pca_train.append(train_data[i])

        docs_pca_labels.append(train_labels[i])
# now let's transform the text data and vectorize it.

pca_vec = vec.fit_transform(docs_pca_train)
#Text Clustering:

#let's start with PCA(principle component anaysis)

from sklearn.decomposition import PCA

variances = []

#now finding the number of components which alteast satisfy half variance

for i in range (1,100,5):

    pca = PCA(i)

    pca.fit(pca_vec.toarray())

    variances.append(pca.explained_variance_ratio_.sum())
#plotting the variances to find the number of components:

import matplotlib.pyplot as plt

plt.figure(figsize = (8, 8))

plt.grid()

x= range(1,100,5)

plt.plot(x, variances, 'b*')

plt.xlabel('number of components')

plt.ylabel('Variance')

plt.title('finding the number of components')

plt.show()
#the number of components obtained are 60

#now performing pca for 60 components

pca = PCA(60)

pca.fit(pca_vec.toarray())

docs_pca = pca.transform(pca_vec.toarray())
#Converting labels into encoded format of int

labelBinarizer = MultiLabelBinarizer()

data_labels_binary = labelBinarizer.fit_transform(docs_pca_labels)

data_labels_encode = data_labels_binary.argmax(axis = 1)

data_labels_encode = data_labels_encode.astype(int)
#let's plot the PCA clusters

import seaborn as sns

plt.figure(figsize = (8, 8 ))

plt.grid()

sns.scatterplot(docs_pca[:, 0], docs_pca[:, 1], hue =data_labels_encode)

plt.show()
from sklearn.cluster import KMeans

k_means = KMeans(5, max_iter =100)

clusters = k_means.fit_predict(pca_vec)
#visualising the k means

plt.figure(figsize = (8,8))

plt.grid()

sns.scatterplot(docs_pca[:, 0], docs_pca[:, 1], hue = clusters)

plt.show()
#let's perform text classification:

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
models = [('multinomial_nb', MultinomialNB()),

          ('logistic', LogisticRegression()),

          ('linear_svc', LinearSVC()),

          ('knn', KNeighborsClassifier(n_neighbors = 6)),

          ('rf', RandomForestClassifier(n_estimators = 6))]
binarizers = MultiLabelBinarizer()

train_labels = binarizers.fit_transform(train_labels)

test_labels = binarizers.transform(test_labels)
train_data = vec.fit_transform(train_data)

test_data = vec.transform(test_data)
print('the shape of train data is:', train_data.shape,'\n test data shape:', test_data.shape)

print('thhe train labels shape is:', train_labels.shape,'\n test labels shape is:', test_labels.shape)
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report

from sklearn.multiclass import OneVsRestClassifier

for model_name, model in models:

    classifier = OneVsRestClassifier(model)

    classifier.fit(train_data, train_labels)

    Y_test = classifier.predict(test_data)

    print(classification_report(Y_test, test_labels))