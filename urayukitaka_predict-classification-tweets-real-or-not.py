import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import re

from collections import Counter

import time

import pickle

import itertools



# Visualization

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

import seaborn as sns



# data preprocessing

from sklearn.model_selection import train_test_split



# Validation

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score



# text preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer



# Kearas

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Activation, Dense, Embedding, LSTM, Dropout, BatchNormalization, Conv1D, Flatten, MaxPooling1D

from keras import utils

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.models import load_model

from keras.optimizers import Adam



# Word2Vec

import gensim
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sample = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train.head()
test.head()
sample.head()
print("train data size:", train.shape)

print("test data size:", test.shape)
print("train_data null count\n", train.isnull().sum())
print("test_data null count\n", test.isnull().sum())
print("train_data data type\n", train.dtypes)
print("test_data data type\n", test.dtypes)
target_cnt = Counter(train["target"])

print(target_cnt)



plt.figure(figsize=(10,6))

plt.bar([str(i) for i in target_cnt.keys()], target_cnt.values())

plt.xlabel("Target flag")

plt.ylabel("Count")

plt.title("target count distribution")
# key 

train_keyw = pd.DataFrame({"keyword":train["keyword"].value_counts().index,

              "Train_count":train["keyword"].value_counts()})

test_keyw = pd.DataFrame({"keyword":test["keyword"].value_counts().index,

              "Test_count":test["keyword"].value_counts()})

keyw = pd.merge(train_keyw, test_keyw, left_on="keyword", right_on="keyword", how="outer").reset_index().fillna(0)

print("keyword sample data shape:", keyw.shape)

keyw.head()
# Visualization

plt.figure(figsize=(30,6))

plt.bar(keyw.keyword, keyw.Train_count, width=0.4)

plt.bar(keyw.keyword, keyw.Test_count, width=0.4)

plt.xlabel("Key word")

plt.xticks(rotation=90)

plt.ylabel("Count")

plt.title("Keyword null count\nTrain_data:{}/Test_data:{}".format(train.keyword.isnull().sum(), test.keyword.isnull().sum()))
# location

train_loc = pd.DataFrame({"location":train["location"].value_counts().index,

              "Train_count":train["location"].value_counts()})

test_loc = pd.DataFrame({"location":test["location"].value_counts().index,

              "Test_count":test["location"].value_counts()})

loc = pd.merge(train_loc, test_loc, left_on="location", right_on="location", how="outer").reset_index().fillna(0)

print("loc sample data shape:", loc.shape)

loc.head()
# Stop word

stop_words = stopwords.words("english")

stemmer = SnowballStemmer("english")



def preprocessing_text(text, stem=False):

    text = re.sub(r",", '', str(text).lower())

    text = re.sub(r"https?:\S", ' ', str(text))

    text = re.sub(r"[^a-zA-Z]+", ' ', str(text)).strip()

    

    tokens = []

    for token in text.split():

        if token not in stop_words:

            if stem:

                tokens.append(stemmer.ste,(token))

            else:

                tokens.append(token)

    return ' '.join(tokens)
# train data set

train["cleaned_text"] = train["text"].apply(lambda x:preprocessing_text(x))
# test data set

test["cleaned_text"] = test["text"].apply(lambda x:preprocessing_text(x))
# fillna

train["keyword"].fillna("Nan_keyw", inplace=True)

train["location"].fillna("Nan_loc", inplace=True)

test["keyword"].fillna("Nan_keyw", inplace=True)

test["location"].fillna("Nan_loc", inplace=True)
# Combine keyword + location to text

train["cleaned_text"] = train["keyword"] + str(" ") + train["location"] +  str(" ") + train["cleaned_text"]

test["cleaned_text"] = test["keyword"] +  str(" ") + test["location"] +  str(" ") + test["cleaned_text"]
# Create train and val data

X = train["cleaned_text"]

y = train["target"]
doc = [w.split() for w in X.values]
# Checking

np.array(doc)[0]
# Word2Vec model

w2v = gensim.models.word2vec.Word2Vec(size = 42, 

                                      window = 8,

                                      alpha = 0.03,

                                      workers = 8)



w2v.build_vocab(doc)
words = w2v.wv.vocab.keys()

print("words size:{}".format(len(words)))
%%time

# Training w2v model

w2v.train(doc, total_examples=len(doc), epochs=64)
# example : like

w2v.most_similar("like")
# example : sad

w2v.most_similar("sad")
# example : nice

w2v.most_similar("nice")
# Create train and val data

X = train["cleaned_text"]

y = train["target"]



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=20)



print("X_train_data size:", len(X_train))

print("X_test_data size:", len(X_val))





# tokenizer and keras setting

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X)



X = pad_sequences(tokenizer.texts_to_sequences(X), maxlen=42)

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=42)

X_val = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=42)
# Data dimension check

print("X_train shape:", X_train.shape)

print("X_val shape:", X_val.shape)
# target data dimension change and check

y_train = np.array(y_train).reshape(-1,1)

y_val = np.array(y_val).reshape(-1,1)



print("y_train shape:", y_train.shape)

print("y_val shape:", y_val.shape)
# Embeddint matrix

embedding_matrix = np.zeros((len(words)+1, 42))

for word, i in tokenizer.word_index.items():

    if word in w2v.wv:

        embedding_matrix[i] = w2v.wv[word]



embedding_l = Embedding(len(words)+1, 42, weights=[embedding_matrix], input_length=42, trainable=False)



# Create keras model

def define_model():

    model = Sequential()

    model.add(embedding_l)

    model.add(Dropout(0.15))

    

    model.add(Dense(256, activation="relu"))

    model.add(Conv1D(filters=4, strides=1, kernel_size=2))

    model.add(BatchNormalization())

    model.add(MaxPooling1D(pool_size=2))

    model.add(Dropout(0.3))

    

    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

    

    model.add(Dense(16, activation="relu"))

    model.add(Dropout(0.2))

    

    model.add(Dense(1, activation='sigmoid'))

    

    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0001), metrics=["accuracy"])

    

    return model
model = define_model()



# callbacks

es = EarlyStopping(monitor="val_loss", patience=20)

ms = ModelCheckpoint("emb_lstm_v1", monitor="val_loss", save_best_only=True, verbose=1)



model.summary()
%%time

# Execute fitting

history = model.fit(X_train, y_train, batch_size=256, epochs=300, validation_data=(X_val, y_val), verbose=1, callbacks=[es,ms])
# Visualization

 

train_loss = history.history["loss"]

val_loss = history.history["val_loss"]

train_acc = history.history["accuracy"]

val_acc = history.history["val_accuracy"]



# Visualization

fig, ax = plt.subplots(1,2,figsize=(20,6))

ax[0].plot(range(len(train_loss)), train_loss, label="train_loss")

ax[0].plot(range(len(val_loss)), val_loss, label="val_loss")

ax[0].set_xlabel("epoch")

ax[0].set_ylabel("loss")

ax[0].legend()



ax[1].plot(range(len(train_acc)), train_acc, label="train_acc")

ax[1].plot(range(len(val_acc)), val_acc, label="val_acc")

ax[1].set_xlabel("epoch")

ax[1].set_ylabel("accuracy")

ax[1].legend()
# Validation of val data

y_pred = load_model("emb_lstm_v1").predict(X_val)
y_prediction = []

for i in range(len(y_pred)):

    y_ = y_pred[i][0]

    if y_ >= 0.5:

        res = 1

    else:

        res = 0

    y_prediction.append(res)
print("accuracy_score\n", accuracy_score(y_val, y_prediction))

print("confusion_matrix\n", confusion_matrix(y_val, y_prediction))

print("classification_report\n", classification_report(y_val, y_prediction))
# Test data preprocessing

X_test = test["cleaned_text"]



X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=42)
y_test_pred = load_model("emb_lstm_v1").predict(X_test)



y_test_prediction = []

for i in range(len(y_test_pred)):

    y_ = y_test_pred[i][0]

    if y_ >= 0.5:

        res = 1

    else:

        res = 0

    y_test_prediction.append(res)
sample["target"] = y_test_prediction
sample["target"].value_counts()
sample.to_csv("submission.csv", index=False)