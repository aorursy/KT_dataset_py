# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Load packages 

import math

import re

import os

import timeit

import tensorflow as tf

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

import logging

import time



from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logging.basicConfig(format='[%(asctime)s %(levelname)8s] %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')



! cp ../input/quora-insincere-questions-classification/*.csv .
class C_LR():

  def __init__(self):

    self.train = None

    self.test = None 

    print("Class LR")



  def load_data(self, train_file='train.csv', test_file='test.csv'):

      """ Load train, test csv files and return pandas.DataFrame

      """

      self.train = pd.read_csv(train_file, engine='python', encoding='utf-8', error_bad_lines=False)

      self.test = pd.read_csv(test_file, engine='python', encoding='utf-8', error_bad_lines=False)

      logging.info('CSV data loaded')

      print("...")

    

    

  def countvectorize(self):

      tv = CountVectorizer()

      tv.fit(self.train.question_text)

      self.vector_train = tv.transform(self.train.question_text)

      self.vector_test  = tv.transform(self.test.question_text)

      print("Train & test text tokenized")



  def run_model(self):

      nb = MultinomialNB()

      labels = self.train.target

      x_train, x_val, y_train, y_val = train_test_split(self.vector_train, labels, test_size=0.2, random_state=23)

      nb.fit(x_train, y_train)

      y_preds = nb.predict(x_val)

      print(f"Accuracy score: {accuracy_score(y_val, y_preds)}")



      y_preds = nb.predict_proba(self.vector_test)

      return y_preds

  

  def save_predictions(self, y_preds):

      sub = pd.read_csv('sample_submission.csv')

      for i in range(len(y_preds)):

        if len(self.test.iloc[i].question_text) <= 10:

            y_preds[i] = 1

                

      sub['prediction'] = y_preds 

    

      sub.to_csv('submission.csv', index=False)

      print('Prediction exported to submisison.csv')



lr = C_LR()

lr.load_data()

# b.train = b.train.sample(100000)

lr.countvectorize()

labels = lr.train.target

x_train, x_val, y_train, y_val = train_test_split(lr.vector_train, labels, test_size=0.2, random_state=2090)



model = LogisticRegression(n_jobs=10, solver='saga', C=0.1, verbose=1)

model.fit(x_train, y_train)

y_preds = model.predict(x_val)



print(f"Accuracy score: {accuracy_score(y_val, y_preds)}")

print(f"Confusion matrix: ") 

print(confusion_matrix(y_val, y_preds))

print("Classificaiton report:\n", classification_report(y_val, y_preds, target_names=["Sincere", "Insincere"]))



y_probs = model.predict_proba(x_val)

y_probs

best_threshold = best_f1 = 0



for i in range(0, 100):

  y2_preds = [1 if e[1] >= i / 100 else 0 for e in y_probs]

  cur_f1 = f1_score(y_val, y2_preds)

  print(i, cur_f1)

  if cur_f1 > best_f1:

    best_f1 = cur_f1

    best_threshold = i / 100



print(f"Best f1 score {best_f1}, best threshold {best_threshold}")

y_probs = model.predict_proba(x_val)

y_probs

y2_preds = [1 if e[1] >= 0.25 else 0 for e in y_probs]



print(f"Confusion matrix: ") 

print(confusion_matrix(y_val, y2_preds))

print("Classificaiton report:\n", classification_report(y_val, y2_preds, target_names=["Sincere", "Insincere"]))



test_proba = model.predict_proba(lr.vector_test)

test_preds = [1 if e[1] >= 0.19 else 0 for e in test_proba]

sub = pd.read_csv(f"sample_submission.csv")

sub['prediction'] = test_preds

sub.to_csv('submission.csv', index=False)
!head -n 10 submission.csv