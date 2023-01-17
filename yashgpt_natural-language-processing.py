import pandas as pd

import numpy as np
df=pd.read_csv('../input/yelp-reviews-dataset/yelp.csv')
df.head()
df.info()
df.describe()
df['text length']=df['text'].apply(len)

df.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
g=sns.FacetGrid(df,col='stars')

g.map(plt.hist,'text length')
sns.boxplot(x=df['stars'],y=df['text length'],palette='rainbow')
sns.countplot(df['stars'])
df1=pd.DataFrame(df.groupby('stars').mean())

df1
df1.corr()
plt.figure(figsize=(10,8))

sns.heatmap(df1.corr(),cmap='coolwarm',annot=True)
yelp_class=df[(df['stars']==1)|(df['stars']==5)]

yelp_class.head()
X=yelp_class.text

y=yelp_class.stars
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()
X=cv.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.30)
from sklearn.naive_bayes import MultinomialNB

nb=MultinomialNB()
nb.fit(X_train,y_train)
pred=nb.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
pipeline=Pipeline([

    ('bow', CountVectorizer()),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier

])
X = yelp_class['text']

y = yelp_class['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))