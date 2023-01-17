%config IPCompleter.greedy=True
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import re
dataset = pd.read_csv('../input/spam.csv',encoding="ISO-8859-1")
dataset.head()
dataset = dataset.rename(columns={"v1":"label","v2":"message"})
dataset = dataset.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
dataset['label'].replace("ham",0,inplace=True)
dataset['label'].replace("spam",1,inplace=True)
#clean up anything other than chars and nums
for msg in dataset['message']:
    msg = re.sub('[^A-Za-z0-9 ]+','',msg)
dataset.head()
#split to test and train data
from sklearn.model_selection import train_test_split
import numpy as np
X_train,X_test,y_train,y_test = train_test_split(dataset['message'],dataset['label'],random_state=42,test_size=0.33)
#If you want to know the shape and stats of dataset. 
#print("-"*20)
# print("Training features shape: ",X_train.shape)
# print("Training labels shape: ",y_train.shape)
# print("Test feature shape: ",X_test.shape)
# print("test label shape: ",y_test.shape)
# print("Count:")
# unique, count = np.unique(y_train,return_counts=True)
# spam_ham = dict(zip(unique,count))
# print((spam_ham[1]/(spam_ham[0]+spam_ham[1])*100," were spam in training dataset"))
# print("In original dataset")
# unique, count = np.unique(dataset['label'],return_counts=True)
# spam_ham = dict(zip(unique,count))
# print((spam_ham[1]/(spam_ham[0]+spam_ham[1])*100," were spam in original dataset"))
# print("-"*20)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidfTransformer = TfidfTransformer(use_idf=True,norm="l1")
X_train_tfIdf = tfidfTransformer.fit_transform(X_train_counts)
clf = BernoulliNB().fit(X_train_tfIdf.toarray(),y_train)
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_tfIdf,y_train)
# transform the test feature in the same form as for training feature
X_test_counts = count_vect.transform(X_test)
X_test_tfIdf = tfidfTransformer.transform(X_test_counts)
predicted_BNB = clf.predict(X_test_tfIdf.toarray())
predicted_LR = log_reg_model.predict(X_test_tfIdf)
# measure accuracy
print("Bernoulli NB: ",accuracy_score(y_test,predicted_BNB,normalize=True))
print("Logistic Regression: ",accuracy_score(y_test,predicted_LR,normalize=True))