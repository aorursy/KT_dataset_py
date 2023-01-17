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

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logging.basicConfig(format='[%(asctime)s %(levelname)8s] %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')



! cp ../input/quora-insincere-questions-classification/*.csv .
# Base class for classifier

class Classifier():

  def __init__(self):

    self.train = None

    self.test = None 

    self.model = None



  def load_data(self, train_file='train.csv', test_file='test.csv'):

      """ Load train, test csv files and return pandas.DataFrame

      """

      self.train = pd.read_csv(train_file, engine='python', encoding='utf-8', error_bad_lines=False)

      self.test = pd.read_csv(test_file, engine='python', encoding='utf-8', error_bad_lines=False)

      logging.info('CSV data loaded')

  

  def countvectorize(self):

      tv = TfidfVectorizer(ngram_range=(1,3), token_pattern=r'\w{1,}',

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1, max_features=5000)

      # cv = CountVectorizer()

      text = pd.concat([self.train.question_text, self.test.question_text])

      tv.fit(text)

      self.vector_train = tv.transform(self.train.question_text)

      self.vector_test  = tv.transform(self.test.question_text)

      logging.info("Train & test text tokenized")



  def build_model(self):

      pass



  def run_model(self):

      # Choose your own classifier: self.model and run it

      logging.info(f"{self.__class__.__name__} starts running.")

      labels = self.train.target

      x_train, x_val, y_train, y_val = train_test_split(self.vector_train, labels, test_size=0.2, random_state=2090)

      self.model.fit(x_train, y_train)

      y_preds = self.model.predict(x_val)

      logging.info(f"Accuracy score: {accuracy_score(y_val, y_preds)}")



      y_preds = self.model.predict(self.vector_test)

      return y_preds



  def save_predictions(self, y_preds):

      sub = pd.read_csv(f"sample_submission.csv")

      sub['prediction'] = y_preds 

      sub.to_csv(f"submission_{self.__class__.__name__}.csv", index=False)

      logging.info('Prediction exported to submisison.csv')

  

  def pipeline(self):

      s_time = time.clock()

      self.load_data()

      self.countvectorize()

      self.build_model()

      self.save_predictions(self.run_model())

      logging.info(f"Program running for {time.clock() - s_time} seconds")



class C_Bayes(Classifier):

  def build_model(self):

      self.model = MultinomialNB()

      return self.model



# Logistic Regression 

class C_LR(Classifier):

  def build_model(self):

      self.model = LogisticRegression(n_jobs=10, solver='saga', C=0.1, verbose=1)

      return self.model



class C_SVM(Classifier):

  def load_data(self, train_file='train.csv', test_file='test.csv'):

      """ Load train, test csv files and return pandas.DataFrame

      """

      self.train = pd.read_csv(train_file, engine='python', encoding='utf-8', error_bad_lines=False)

      self.train = self.train.sample(10000)

      self.test = pd.read_csv(test_file, engine='python', encoding='utf-8', error_bad_lines=False)

      logging.info('CSV data loaded')



  def build_model(self):

      self.model = svm.SVC()

      return self.model



class C_Ensemble(Classifier):

  def ensemble(self):

      s_time = time.perf_counter()

      self.load_data()

      self.countvectorize()



      nb = MultinomialNB()

      lr = LogisticRegression(n_jobs=10, solver='saga', C=0.1, verbose=1)

      svc = svm.SVC()



      all_preds = [0] * self.test.shape[0]

      for m in (nb, lr, svc):

          self.model = m

          if m == svc: 

              self.load_data()

              self.train = self.train.sample(10000)

              self.countvectorize()

          all_preds += self.run_model()

      

      all_preds = [1 if p > 0 else 0 for p in all_preds]

      self.save_predictions(all_preds)

      logging.info(f"Program running for {time.perf_counter() - s_time} seconds")



  

C_Ensemble().ensemble()     

! cp submission_C_Ensemble.csv submission.csv
def batch_nb():

  all_preds = []

  b = C_Bayes()

  b.load_data()

  full_train = b.train



  for _ in range(10):

    b.train = full_train.sample(1000000)

    b.countvectorize()

    b.build_model()

    y_preds = b.run_model()

    all_preds = all_preds + y_preds if len(all_preds) > 0 else y_preds



  all_preds = [1 if p >=1  else 0 for p in all_preds]

  b.save_predictions(all_preds)



# batch_nb()
# ! cp submission_C_Bayes.csv submission.csv

# ! ls
