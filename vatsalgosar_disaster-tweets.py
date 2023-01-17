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
import matplotlib.pyplot as plt



from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn import decomposition, ensemble



import pandas, xgboost, numpy, textblob, string

from keras.preprocessing import text, sequence

from keras import layers, models, optimizers



from nltk.stem import PorterStemmer 

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



import re
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
train_df.shape
train_df.head()
print(train_df[train_df['target'] == 0]["text"].values[1])

print(train_df[train_df['target'] == 1]["text"].values[1])
fig = plt.figure(figsize=(4,3))

train_df.groupby('target').target.count().plot.bar(ylim=0)

plt.show()
stopwords = set(stopwords.words('english') + list(string.punctuation))

porter_stemmer = PorterStemmer() 
train_df['keyword'] = train_df['keyword'].fillna('')

train_df['location'] = train_df['location'].fillna('')
def preprocess(text):

    text = text.lower()

    text = " ".join([porter_stemmer.stem(word) for word in word_tokenize(text) if word not in stopwords])

    text = re.sub('\S*@\S*\s?', '', text)

    text = " ".join(re.split("[^a-zA-Z0-9]*", text))

    text = re.sub('[0-9]+', ' ', text)

    text = re.sub('\s+', ' ', text)

    return text
preprocess("Forest fire near La Ronge Sask. Canada")
train_df['updated_text'] = [preprocess(text) for text in train_df['text']]
train_df.head()
# split the dataset into training and validation datasets 

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_df['updated_text'], train_df['target'])



# label encode the target variable 

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)

valid_y = encoder.fit_transform(valid_y)
len(train_y)
len(train_x)
# create a count vectorizer object 

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')

count_vect.fit(train_df['updated_text'])



# transform the training and validation data using count vectorizer object

xtrain_count =  count_vect.transform(train_x)

xvalid_count =  count_vect.transform(valid_x)
# word level tf-idf

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(train_df['updated_text'])

xtrain_tfidf =  tfidf_vect.transform(train_x)

xvalid_tfidf =  tfidf_vect.transform(valid_x)



# ngram level tf-idf 

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram.fit(train_df['updated_text'])

xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)

xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)



# characters level tf-idf

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)

tfidf_vect_ngram_chars.fit(train_df['updated_text'])

xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 

xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):

    # fit the training dataset on the classifier

    classifier.fit(feature_vector_train, label)

    

    # predict the labels on validation dataset

    predictions = classifier.predict(feature_vector_valid)

    

    if is_neural_net:

        predictions = predictions.argmax(axis=-1)

    

    return metrics.accuracy_score(predictions, valid_y)
X_train, X_test, y_train, y_test = train_test_split(train_df['updated_text'], train_df['target'], random_state = 0)

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)
# Naive Bayes on Count Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)

print("NB, Count Vectors: " + str(accuracy))



# Naive Bayes on Word Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)

print("NB, WordLevel TF-IDF: " + str(accuracy)) 



# Naive Bayes on Ngram Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print("NB, N-Gram Vectors: " + str(accuracy))



# Naive Bayes on Character Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

print("NB, CharLevel Vectors: " + str(accuracy))
train_x = train_df['updated_text']

train_y = train_df['target']
# word level tf-idf

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(train_df['updated_text'])

xtrain_tfidf =  tfidf_vect.transform(train_x)
naive_bayes_classifier_model = MultinomialNB().fit(xtrain_tfidf, train_y)
print(naive_bayes_classifier_model.predict(tfidf_vect.transform(["Forest fire near La Ronge Sask. Canada"])))
print(naive_bayes_classifier_model.predict(tfidf_vect.transform(["I love fruits"])))
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission.head()
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
test_df['updated_text'] = [preprocess(text) for text in test_df['text']]
test_vectors = tfidf_vect.transform(test_df["updated_text"])
sample_submission["target"] = naive_bayes_classifier_model.predict(test_vectors)
sample_submission.head()
sample_submission.to_csv("/kaggle/working/submission3.csv", index=False)