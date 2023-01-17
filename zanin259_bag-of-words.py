import warnings



warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
# We load all the required data sets.

train_lab=pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv',delimiter='\t',quoting=3)



train_unlab=pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv',error_bad_lines=False,delimiter='\t')



test=pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/testData.tsv',delimiter='\t',quoting=3)



submission=pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/sampleSubmission.csv')
train_lab.head()
train_lab.info()
train_unlab.head()
train_unlab.info()
test.head()
test.info()
# We drop the id columns from the dataset

train_lab=train_lab.drop(columns=['id'])
# We create a function to review all the words

def review_word(raw_review):

    review=BeautifulSoup(raw_review).get_text()

    review=re.sub('[^a-zA-Z]',' ',review)

    review=review.lower()

    review=review.split()

    ps=PorterStemmer()

    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review=' '.join(review)

    return review
corpus=[]



for i in range(0,25000):

    corpus.append(review_word(train_lab['review'][i]))
# We use a count vectorizer to limit the vocabulary to 5000

tv=TfidfVectorizer(max_features=5000,analyzer='word')
# We split the data into X and y dataset

X=tv.fit_transform(corpus).toarray()



y=train_lab['sentiment']
# We split the dataset into train and test dataset

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# We perform the naive bayes classifier on the model



classifier_nb=GaussianNB()



classifier_nb.fit(X_train,y_train)



y_pred_nb=classifier_nb.predict(X_test)
# We find the accuracy of the model using the classifcation report 



report=classification_report(y_pred_nb,y_test)



print('the classification report is \n{}'.format(report))
# We perform the random forest classifier on the model



classifier_rf=RandomForestClassifier(random_state=0)



classifier_rf.fit(X_train,y_train)



y_pred_rf=classifier_rf.predict(X_test)
# We find the accuracy of the model using the classifcation report 



report=classification_report(y_pred_rf,y_test)



print('the classification report is \n{}'.format(report))
# We perform the Logistic regression on the model



classifier_lr=LogisticRegression(random_state=0)



classifier_lr.fit(X_train,y_train)



y_pred_lr=classifier_lr.predict(X_test)
# We find the accuracy of the model using the classifcation report 



report=classification_report(y_pred_lr,y_test)



print('the classification report is \n{}'.format(report))
# We perform the Linear SVC on the model



classifier_svc=LinearSVC(random_state=0)



classifier_svc.fit(X_train,y_train)



y_pred_svc=classifier_svc.predict(X_test)
# We find the accuracy of the model using the classifcation report 



report=classification_report(y_pred_svc,y_test)



print('the classification report is \n{}'.format(report))
# We create a new tfidf vectorizer to improve the accuracy

tv1=TfidfVectorizer(max_features=5000,analyzer='word',ngram_range=(1,2))
# We split the data into X and y dataset

X1=tv1.fit_transform(corpus).toarray()



y1=train_lab['sentiment']
X_train1,X_test1,y_train1,y_test1=train_test_split(X1,y1,test_size=0.2,random_state=0)
# We perform the Logistic regression on the model



classifier_lr1=LogisticRegression(random_state=0)



classifier_lr1.fit(X_train1,y_train1)



y_pred_lr1=classifier_lr1.predict(X_test1)
# We find the accuracy of the model using the classifcation report 



report=classification_report(y_pred_lr1,y_test1)



print('the classification report is \n{}'.format(report))
# We review the words in the test dataset



corpus_test=[]



for i in range(0,25000):

    corpus_test.append(review_word(test['review'][i]))
# We perform the tfidf vectoriser on the test dataset

test=tv.transform(corpus_test).toarray()
# We use the model to predict

test=classifier_lr.predict(test)



test=pd.DataFrame(test)
# We add the test prediction into the dataset

submission['sentiment']=test
submission
submission.to_csv('Submission.csv',index=False)