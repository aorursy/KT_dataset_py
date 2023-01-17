import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
%matplotlib inline
train_dataset = pd.read_csv('../input/train.tsv',sep="\t")
test_dataset=pd.read_csv('../input/test.tsv',sep="\t")
train_dataset.head()
test_dataset.head()
from nltk.corpus import stopwords
stopwords.words('english')
test_dataset.shape
train_dataset.shape
corpus=[]
for i in range(0,156060):
    review=re.sub('[^a-zA-Z]',' ',train_dataset['Phrase'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
corpus
test_corpus=[]
for j in range(0,66292):
    review=re.sub('[^a-zA-Z]',' ',test_dataset['Phrase'][j])
    review=review.lower()
    review=review.split()
    ps1=PorterStemmer()
    review=[ps1.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    test_corpus.append(review)
len(corpus)
len(test_corpus)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1000)
cv1=CountVectorizer(max_features=1000)
X=cv.fit_transform(corpus).toarray()
Y_new=cv1.fit_transform(test_corpus).toarray()
Y_new.shape
y=train_dataset.iloc[:,3].values
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X,y,test_size=0.02,random_state=0)
x_test.shape
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)
y_pred.shape
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
new_y_pred=classifier.predict(Y_new)
new_y_pred
sub_file=pd.read_csv('../input/sampleSubmission.csv')
sub_file.head()
sub_file.Sentiment=new_y_pred
sub_file.to_csv('submission.csv',index=False)
