import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
df = pd.read_csv('../input/review-data/hello', sep='\t', names=['likes','text'])
df.head(10)
#TFIDF vectorization

stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset )
y=df.likes
x=vectorizer.fit_transform(df.text)
y.shape
x.shape
X_train,X_test,Y_train,Y_test=train_test_split(x, y, random_state=42)
clf=naive_bayes.MultinomialNB()
clf.fit(X_train,Y_train)
roc_auc_score(Y_test, clf.predict_proba(X_test)[:,1])
hold = pd.read_csv('../input/test-input/test', sep='\t')
movie_reviews_array = np.array(hold) 
for i in movie_reviews_array[:20]:
 print (i)
for i in movie_reviews_array[:20]:
 movie_reviews_vector = vectorizer.transform(i)
 print (clf.predict(movie_reviews_vector))