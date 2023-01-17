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

import smart_open



from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logging.basicConfig(format='[%(asctime)s %(levelname)8s] %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')



import keras

from keras.models import Model

from keras import Input, layers



from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Sequential

from keras.layers import Flatten, Dense, Embedding, Dropout, LSTM, GRU, Bidirectional

from keras.utils import to_categorical

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import gensim.downloader as api



# Get data

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

      train = pd.read_csv(train_file, engine='python', encoding='utf-8', error_bad_lines=False)

      self.test = pd.read_csv(test_file, engine='python', encoding='utf-8', error_bad_lines=False)

      self.train = pd.concat([train[train.target == 1].sample(80_000), train[train.target==0].sample(200_000)])

      logging.info('CSV data loaded')

  

  def countvectorize(self):

      tv = TfidfVectorizer(ngram_range=(1,3), token_pattern=r'\w{1,}',

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1, max_features=5000)

#       tv = CountVectorizer()

      tv.fit(self.train.question_text)

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

      logging.info(f"Confusion matrix: ") 

      print(confusion_matrix(y_val, y_preds))

      print("Classificaiton report:\n", classification_report(y_val, y_preds, target_names=["Sincere", "Insincere"]))

      # y_preds = self.model.predict(self.vector_test)

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

      self.model = LogisticRegression(n_jobs=10, solver='lbfgs', C=0.1, verbose=1)

      return self.model



class C_SVM(Classifier):

  def load_data(self, train_file='train.csv', test_file='test.csv'):

      """ Load train, test csv files and return pandas.DataFrame

      """

      self.train = pd.read_csv(train_file, engine='python', encoding='utf-8', error_bad_lines=False)

      self.train = self.train.sample(100000)

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





class Helper():

    def locate_threshold(self, model, x_val, y_val):

        y_probs = model.predict(x_val, batch_size=1024, verbose=1)

        best_threshold = best_f1 = pre_f1 = 0

        history = []



        for i in np.arange(0.01, 1, 0.01):

          if len(y_probs[0]) >= 2:

              y2_preds = [1 if e[1] >= i else 0 for e in y_probs]

          else:

              y2_preds = (y_probs > i).astype(int)



          cur_f1 = f1_score(y_val, y2_preds)

          history.append((i, cur_f1))

          symbol = '+' if cur_f1 >= pre_f1 else '-'

          print("Threshold {:6.4f}, f1_score: {:<0.8f}  {} {:<0.6f} ".format(i, cur_f1, symbol, abs(cur_f1 - pre_f1)))

          pre_f1 = cur_f1



          if cur_f1 >= best_f1:

              best_f1 = cur_f1

              best_threshold = i



        print(f"Best f1 score {best_f1}, best threshold {best_threshold}")

        plt.xlabel('Threshold')

        plt.ylabel('f1_score')

        plt.plot(*zip(*history))



        return best_threshold
class C_NN(Classifier):

    def __init__(self, max_features=100000, embed_size=128, max_len=300):

        self.max_features=max_features

        self.embed_size=embed_size

        self.max_len=max_len

    

    def tokenize_text(self, text_train, text_test):

        '''@para: max_features, the most commenly used words in data set

        @input are vector of text

        '''

        tokenizer = Tokenizer(num_words=self.max_features)

        text = pd.concat([text_train, text_test])

        tokenizer.fit_on_texts(text)



        sequence_train = tokenizer.texts_to_sequences(text_train)

        tokenized_train = pad_sequences(sequence_train, maxlen=self.max_len)

        logging.info('Train text tokeninzed')



        sequence_test = tokenizer.texts_to_sequences(text_test)

        tokenized_test = pad_sequences(sequence_test, maxlen=self.max_len)

        print('Test text tokeninzed')

        return tokenized_train, tokenized_test, tokenizer

      

    def build_model(self):

        dropout = 0.2

        model = Sequential()

        model.add(Embedding(self.max_features, self.embed_size, input_length=self.max_len))

        model.add(Bidirectional(GRU(64, return_sequences=True)))

        model.add(Bidirectional(GRU(64, return_sequences=True)))



        model.add(Flatten())



        model.add(Dense(32, activation='relu'))

        model.add(Dropout(dropout))

        model.add(Dense(32, activation='relu'))

        model.add(Dropout(dropout))

        

        model.add(Dense(1, activation='sigmoid'))

        self.model = model



        return self.model



    def run(self, x_train, y_train):

        checkpoint = ModelCheckpoint('weights_base_best.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        early = EarlyStopping(monitor="val_acc", mode="max", patience=5)



        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

        X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=2020)

        BATCH_SIZE = max(16, 2 ** int(math.log(len(X_tra) / 100, 2)))

        print(f"Batch size is set to {BATCH_SIZE}")

        history = self.model.fit(X_tra, y_tra, epochs=2, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), \

                              callbacks=[checkpoint, early], verbose=1)



        y_pred = self.model.predict(X_val, batch_size=64, verbose=1)

        y_pred_bool = np.argmax(y_pred, axis=1)

        print(classification_report(y_val, y_pred_bool))

        return history



    

# c = C_NN(max_features=50000, embed_size=300, max_len=250)

# c.load_data()

# # c.train = c.train.sample(10000)

# # c.test = c.test.sample(1000)

# vector_train, vector_test, _ = c.tokenize_text(c.train.question_text, c.test.question_text)

# model = c.build_model()

# print("DONE")
class Kerasapi():

    def __init__(self):

        self.embed_size=300

        self.max_features=50000

        self.max_len=100

    

    def build_model(self):

        text_input = Input(shape=(self.max_len, ))

        embed_text = layers.Embedding(self.max_features, self.embed_size)(text_input)

        

        branch_a = layers.Bidirectional(layers.GRU(64, return_sequences=True))(embed_text)

        branch_b = layers.GlobalMaxPool1D()(branch_a)

        branch_c = layers.Dense(64, activation='relu')(branch_b)

        branch_d = layers.Dropout(0.3)(branch_c)

        branch_c = layers.Dense(64, activation='relu')(branch_b)

        branch_d = layers.Dropout(0.3)(branch_c)

        branch_z = layers.Dense(1, activation='sigmoid')(branch_d)

        

        model = Model(inputs=text_input, outputs=branch_z)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        

        return model



c = C_NN(max_features=50000, embed_size=300, max_len=100)

c.load_data()

vector_train, vector_test, _ = c.tokenize_text(c.train.question_text, c.test.question_text)



model = Kerasapi().build_model()

print(">> model was built.")
X_tra, X_val, y_tra, y_val = train_test_split(vector_train, c.train.target, test_size=0.1, random_state=0)

print(">> train test split DONE.")



mc = keras.callbacks.ModelCheckpoint('best.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

es = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)



history = model.fit(X_tra, y_tra, epochs=30, batch_size=1024, callbacks=[mc, es], validation_data=(X_val, y_val), verbose=1)



# y_pred = c.model.predict(X_val, batch_size=1024, verbose=1)

print(">> train completed.")
model.load_weights('best.h5')

best_threshold = Helper().locate_threshold(model, X_val, y_val)
sub = pd.read_csv(f"sample_submission.csv")

y_preds = model.predict(vector_test, batch_size=1024, verbose=1)

y_preds = (y_preds > best_threshold).astype(int)

sub.prediction = y_preds

sub.to_csv('submission.csv', index=False)