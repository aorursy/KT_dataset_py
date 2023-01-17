# importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import seaborn as sns

import json



! cp ../input/indian-products-on-amazon/amazon_vfl_reviews.csv .

json_file = 'amazon_vfl_reviews.csv'

df = pd.read_csv(json_file)

df.info()
df.head()
for tp in df.groupby('name')['asin']:

    if len(tp[1].unique()) > 1:

        print(tp[0], tp[1].unique())
f, ax = plt.subplots(1, 2, figsize=(18,8))

cs = ['r', 'dodgerblue', 'orange', 'green', 'pink']

df['rating'].value_counts().plot(kind='pie', autopct='%2.2f%%', ax=ax[0], colors=cs)

df['rating'].value_counts().plot(kind='barh', ax=ax[1], color=cs)

ax[0].set_title('Share of rating (pie)')

ax[0].set_ylabel('Rating Share')

ax[1].set_title('Share of ratin(bar)')

plt.show()
df[df.review.isna()]
# replace NaN-valued review with word NULL

df.fillna({'review': 'NULL'}, inplace=True)
from wordcloud import WordCloud

import os

from PIL import Image

import urllib



# Control the font for our wordcloud

if not os.path.exists('Comfortaa-Regular.ttf'):

    urllib.request.urlretrieve('http://git.io/JTqLk', 'Comfortaa-Regular.ttf')



if not os.path.exists('cloud.png'):

    urllib.request.urlretrieve('http://git.io/JTORU', 'cloud.png')

    

text = ' '.join(str(t) for t in df.review)

mask = np.array(Image.open('cloud.png'))

wc = WordCloud(max_words=100, background_color='white', 

              font_path='./Comfortaa-Regular.ttf', mask=mask,

#                 max_font_size=100,

              width=mask.shape[1], height=mask.shape[0]).generate(text)



plt.figure(figsize=(24, 12))

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.show()
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score





class Bayes:

    def _pipeline(self, df):

        cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

        review = df[['review']]

        Xtrain, Xtest, ytrain, ytest = train_test_split(review, df.rating, random_state=2)

        cv.fit(pd.concat([Xtrain.review, Xtest.review]))

        Xtrain = cv.transform(Xtrain.review)

        Xtest  = cv.transform(Xtest.review)



        model = MultinomialNB()

        model.fit(Xtrain, ytrain)

        

        ypred = model.predict(Xtest)

        print("Bayes model accuracy score: ", accuracy_score(ytest, ypred))

                             

Bayes()._pipeline(df)
from xgboost import XGBClassifier



class Xgb:

    def _pipeline(self, df):

        cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

        review = df[['review']]

        Xtrain, Xtest, ytrain, ytest = train_test_split(review, df.rating, random_state=2)

        cv.fit(pd.concat([Xtrain.review, Xtest.review]))

        Xtrain = cv.transform(Xtrain.review)

        Xtest  = cv.transform(Xtest.review)



        model = XGBClassifier()

        model.fit(Xtrain, ytrain)

        

        ypred = model.predict(Xtest)

        print("Xgboost classifier accuracy score: ", accuracy_score(ytest, ypred))

                             

Xgb()._pipeline(df)
### import keras 

from keras import layers, Input

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Sequential, Model, load_model

from keras.layers import Flatten, Dense, Embedding, Dropout, LSTM, GRU, Bidirectional

from keras.utils import to_categorical

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import gensim.downloader as api

import logging

import math

# from tqdm.notebook import tqdm

import tensorflow_hub as hub



class Classifier():

  def __init__(self):

    self.train = None

    self.test = None 

    self.model = None

    

  def load_data(self, df):

      """ Load train, test csv files and return pandas.DataFrame

      """

      self.train, self.test = train_test_split(df, test_size=0.2)

      self.train.rename({'review': 'text', 'rating': 'target'}, axis='columns', inplace=True)

      self.test.rename({'review': 'text', 'rating': 'target'}, axis='columns', inplace=True)



  

  def save_predictions(self, y_preds):

      sub = pd.read_csv(f"sampleSubmission.csv")

      sub['Sentiment'] = y_preds 

      sub.to_csv(f"submission_{self.__class__.__name__}.csv", index=False)

      logging.info(f'Prediction exported to submission_{self.__class__.__name__}.csv')

  



class C_NN(Classifier):

    def __init__(self, max_features=10000, embed_size=128, max_len=300):

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

        logging.info('Test text tokeninzed')

        return tokenized_train, tokenized_test, tokenizer

      

    def build_model(self, embed_matrix=[]):

        text_input = Input(shape=(self.max_len, ))

        embed_text = layers.Embedding(self.max_features, self.embed_size)(text_input)

        if len(embed_matrix) > 0:

            embed_text = layers.Embedding(self.max_features, self.embed_size, \

                                          weights=[embed_matrix], trainable=False)(text_input)

            

        branch_a = layers.Bidirectional(layers.GRU(32, return_sequences=True))(embed_text)

        branch_b = layers.GlobalMaxPool1D()(branch_a)



        x = layers.Dense(64, activation='relu')(branch_b)

        x = layers.Dropout(0.2)(x)



        x = layers.Dense(32, activation='relu')(branch_b)

        x = layers.Dropout(0.2)(x)

        branch_z = layers.Dense(6, activation='softmax')(x)

        

        model = Model(inputs=text_input, outputs=branch_z)

        self.model = model



        return model

        

    def embed_word_vector(self, word_index, model='glove-wiki-gigaword-100'):

        glove = api.load(model) # default: wikipedia 6B tokens, uncased

        zeros = [0] * self.embed_size

        matrix = np.zeros((self.max_features, self.embed_size))

          

        for word, i in word_index.items(): 

            if i >= self.max_features or word not in glove: continue # matrix[0] is zeros, that's also why >= is here

            matrix[i] = glove[word]



        logging.info('Matrix with embedded word vector created')

        return matrix



    def run(self, x_train, y_train):

        checkpoint = ModelCheckpoint('weights_base_best.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        early = EarlyStopping(monitor="val_acc", mode="max", patience=3)



        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

        X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=2020)

        BATCH_SIZE = max(16, 2 ** int(math.log(len(X_tra) / 100, 2)))

        logging.info(f"Batch size is set to {BATCH_SIZE}")

        history = self.model.fit(X_tra, y_tra, epochs=30, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), \

                              callbacks=[checkpoint, early], verbose=0)



        return history





c = C_NN(max_features=10000, embed_size=300, max_len=300)

c.load_data(df)  

labels = to_categorical(c.train.target, num_classes=6)

labels



vector_train, vector_test, tokenizer = c.tokenize_text(c.train.text, c.test.text)

embed = c.embed_word_vector(tokenizer.word_index, 'word2vec-google-news-300')

c.build_model(embed_matrix=embed)

c.run(vector_train, labels)

model = load_model('weights_base_best.hdf5')

y_preds = model.predict(vector_test)

final = np.argmax(y_preds, axis=1)

print('CNN accuracy score is', accuracy_score(c.test.target, final))
