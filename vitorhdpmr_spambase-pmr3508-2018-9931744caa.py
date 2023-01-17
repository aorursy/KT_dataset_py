import os
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt 
import numpy as np
os.listdir('../input/spambase')
train = pd.read_csv('../input/spambase/train_data (2).csv')
test = pd.read_csv('../input/spambase/test_features.csv')
train.head()
train.describe(include = [np.number])
train_ = train.iloc[:,:54]
train_.mean().plot(kind = 'bar')
train_.std().plot(kind = 'bar')
train_.corr()
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
palavras = ['word_freq_make','word_freq_address','word_freq_all','word_freq_internet','word_freq_mail','word_freq_people','word_freq_free','word_freq_650','word_freq_lab','char_freq_(','char_freq_;','word_freq_edu','word_freq_table','char_freq_$','char_freq_#','word_freq_cs','word_freq_technology']
Xtrain = train[palavras]
Ytrain = train['ham']
Xtest = test[palavras]
bnb.fit(Xtrain, Ytrain)
pred = bnb.predict(Xtest)
Xid = test['Id']
array = np.vstack((Xid, pred)).T
final = pd.DataFrame(columns=['id', 'ham'], data=array)
final.to_csv('resultsHam.csv', index = False)