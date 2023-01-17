import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from gensim.models import doc2vec
from collections import namedtuple
from sklearn.model_selection import train_test_split
from time import time
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import logging
#loading training data
df = pd.read_csv('../input/train_tickets.csv')
X_train = df['Title']
y_train  = df['class']
#Feature extraction using count vectorization and tfidf.
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(X_train)
freq_term_matrix = count_vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.transform(freq_term_matrix)
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf', linear_model.LogisticRegression(C=1e5)),
])
t0 = time()
pipeline.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print()


pipeline.predict(["i need safe mode"])

result = pipeline.predict(["how to fix lan cable"])
print(result) #predicting the class assigned to answer here.
