# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# DataFrame

import pandas as pd



# Matplot

import matplotlib.pyplot as plt

%matplotlib inline



# Scikit-learn

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import TfidfVectorizer



# Keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM

from keras import utils

from keras.callbacks import ReduceLROnPlateau, EarlyStopping



# nltk

import nltk

from nltk.corpus import stopwords

from  nltk.stem import SnowballStemmer



# Word2vec

import gensim



# Utility

import re

import numpy as np

import os

from collections import Counter

import logging

import time

import pickle

import itertools



# Set log

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nltk.download('stopwords')
# DATASET

#DATASET_COLUMNS = ["textID", "text", "selected_text", "sentiment"]

DATASET_ENCODING = "ISO-8859-1"

TRAIN_SIZE = 0.8



# TEXT CLENAING

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"



# WORD2VEC 

W2V_SIZE = 300

W2V_WINDOW = 7

W2V_EPOCH = 32

W2V_MIN_COUNT = 10



# KERAS

SEQUENCE_LENGTH = 300

EPOCHS = 8

BATCH_SIZE = 1024



# SENTIMENT

POSITIVE = "POSITIVE"

NEGATIVE = "NEGATIVE"

NEUTRAL = "NEUTRAL"

SENTIMENT_THRESHOLDS = (0.4, 0.7)



# EXPORT

KERAS_MODEL = "model.h5"

WORD2VEC_MODEL = "model.w2v"

TOKENIZER_MODEL = "tokenizer.pkl"

ENCODER_MODEL = "encoder.pkl"
dataset_filename = "/kaggle/input/tweet-sentiment-extraction/train.csv"

test = "/kaggle/input/tweet-sentiment-extraction/test.csv"

sub_sample = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

dataset_path = os.path.join("..","/kaggle/input",dataset_filename)

test_path = os.path.join("..","/kaggle/input",test)

sub_path = os.path.join("..","/kaggle/input",sub_sample)

print("Open file:", dataset_path)

print("Open file:", test)

print("Open file:", sub_sample)

df = pd.read_csv(dataset_path, encoding =DATASET_ENCODING , engine = 'python')

test_df = pd.read_csv(test, encoding =DATASET_ENCODING, engine = 'python')

submission = pd.read_csv(sub_path)
print("Dataset size:", len(df))
df.head(5)
target_cnt = Counter(df.sentiment)



plt.figure(figsize=(16,8))

plt.bar(target_cnt.keys(), target_cnt.values())

plt.title("Dataset labels distribuition")
stop_words = stopwords.words("english")

stemmer = SnowballStemmer("english")
def preprocess(text, stem=False):

    # Remove link,user and special characters

    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()

    tokens = []

    for token in text.split():

        if token not in stop_words:

            if stem:

                tokens.append(stemmer.stem(token))

            else:

                tokens.append(token)

    return " ".join(tokens)
%%time

df.text = df.text.apply(lambda x: preprocess(x))

df.selected_text = df.selected_text.apply(lambda x: preprocess(x))
print("TRAIN size:", len(df))

print("TEST size:", len(test_df))
%%time

documents = [_text.split() for _text in df.text]

documents1 = [_text.split() for _text in df.selected_text]
w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 

                                            window=W2V_WINDOW, 

                                            min_count=W2V_MIN_COUNT, 

                                            workers=8)
w2v_model.build_vocab(documents)
words = w2v_model.wv.vocab.keys()

vocab_size = len(words)

print("Vocab size", vocab_size)
%%time

w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
w2v_model.most_similar("love")
%%time

tokenizer = Tokenizer()

tokenizer.fit_on_texts(df.text)



vocab_size = len(tokenizer.word_index) + 1

print("Total words", vocab_size)
%%time

train = pad_sequences(tokenizer.texts_to_sequences(df.text), maxlen=SEQUENCE_LENGTH)

test = pad_sequences(tokenizer.texts_to_sequences(test_df.text), maxlen=SEQUENCE_LENGTH)
labels = df.sentiment.unique().tolist()

#labels.append(NEUTRAL)

labels
encoder = LabelEncoder()

encoder.fit(df.sentiment.tolist())



y_train = encoder.transform(df.sentiment.tolist())

y_test = encoder.transform(test_df.sentiment.tolist())



y_train = y_train.reshape(-1,1)

y_test = y_test.reshape(-1,1)



print("y_train",y_train.shape)

print("y_test",y_test.shape)
print("train", train.shape)

print("y_train", y_train.shape)

print()

print("test", test_df.shape)

print("y_test", y_test.shape)
y_train[:10]
embedding_matrix = np.zeros((vocab_size, W2V_SIZE))



for word, i in tokenizer.word_index.items():

    

      if word in w2v_model.wv:

        embedding_matrix[i] = w2v_model.wv[word]

print(embedding_matrix.shape)
embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length= 300, trainable=False)
model = Sequential()

model.add(embedding_layer)

model.add(Dropout(0.5))

model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(3, activation='softmax'))



model.summary()
model.compile(loss='sparse_categorical_crossentropy',

              optimizer="adam",

              metrics=['accuracy'])
callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),

              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
%%time

history = model.fit(train, y_train,

                    batch_size=BATCH_SIZE,

                    epochs=EPOCHS,

                    validation_split=0.1,

                    verbose=1,

                    callbacks=callbacks)
%%time

score = model.evaluate(test, y_test, batch_size=BATCH_SIZE)

print()

print("ACCURACY:",score[1])

print("LOSS:",score[0])
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

 

epochs = range(len(acc))

 

plt.plot(epochs, acc, 'b', label='Training acc')

plt.plot(epochs, val_acc, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

 

plt.figure()

 

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

 

plt.show()
def decode_sentiment(score, include_neutral=True):

    if include_neutral:        

        label = NEUTRAL

        if score <= SENTIMENT_THRESHOLDS[0]:

            label = NEGATIVE

        elif score >= SENTIMENT_THRESHOLDS[1]:

            label = POSITIVE



        return label

    else:

        return NEGATIVE if score < 0.5 else POSITIVE     
def predict(text, include_neutral=True):

    start_at = time.time()

    # Tokenize text

    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)

    # Predict

    score = model.predict([x_test])[0]

    #print(score)

    # Decode sentiment

    label = decode_sentiment(score, include_neutral=include_neutral)



    return {"label": label, "score": float(score),

       "elapsed_time": time.time()-start_at}  
predict("I love the music")