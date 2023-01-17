from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from kor_preprocessing import encode_str

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras



tf.version.VERSION
nRowsRead = None # specify 'None' if want to read whole file

# hate_speech_data.csv has 2000 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/korean-extremist-website-womad-hate-speech-data/hate_speech_data.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'hate_speech_data.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
X, label = df1.iloc[:,1], df1.iloc[:,2]



X.head()
label.head()
from sklearn.utils import shuffle

train_size = 1900

X_train, X_test = shuffle(X[:train_size]), shuffle(X[train_size:])

label_train, label_test = shuffle(label[:train_size]), shuffle(label[train_size:])



# https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)



from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



from sklearn.naive_bayes import MultinomialNB

text_clf = MultinomialNB().fit(X_train_tfidf, label_train)



predicted = text_clf.predict(tfidf_transformer.transform(count_vect.transform(X_test)))

print("Naive_Bayes baseline: {}".format(np.mean(predicted == label_test)))
ds = tf.data.Dataset.from_tensor_slices((X, label))

for value in ds.take(5):

    print(value)
ds_X = ds.map(lambda x, Y: x)

ds_Y = ds.map(lambda x, Y: Y)



def encode(str_tensor, length=20):

    s = str_tensor.numpy().decode('UTF-8')

    try:

        return np.array(encode_str(s, length))

    except Exception as ex:

        print(s, ex)

        raise ex



def tf_encode(str_tensor, length=20):

    return tf.py_function(encode,

                       [str_tensor, length],

                       [tf.int64])



ds_X = ds_X.map(tf_encode)

for value in ds_X.take(5):

    print(value)
TEST_SIZE = 100

BATCH = 25



ds_X_test = ds_X.take(TEST_SIZE)

ds_Y_test = ds_Y.take(TEST_SIZE)



ds_X_train = ds_X.skip(TEST_SIZE)

ds_Y_train = ds_Y.skip(TEST_SIZE)



ds_test = tf.data.Dataset.zip((ds_X_test, ds_Y_test)) 

ds_train = tf.data.Dataset.zip((ds_X_train, ds_Y_train))



#BATCH x length x ENCODE

ds_test = ds_test.repeat().batch(BATCH, True)

ds_train = ds_train.repeat().batch(BATCH, True)



for value in ds_test.take(2):

    print(value)

for value in ds_train.take(2):

    print(value)
simple_model = keras.Sequential()



# embedding 25 x length x 256

simple_model.add(keras.layers.Dense(256, input_shape=(20, 120)))

# biGRU 25 x 256

simple_model.add(keras.layers.Bidirectional(keras.layers.GRU(256)))

# dense output 25 x 1

simple_model.add(keras.layers.Dense(1, activation='sigmoid'))



simple_model.compile(loss='binary_crossentropy', metrics=['acc'])

simple_model.summary()



EPOCHS = 100

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

simple_model.fit(x=ds_train.repeat().as_numpy_iterator(),

                 validation_data=ds_test.repeat().as_numpy_iterator(),

                 batch_size=BATCH,

                 epochs=EPOCHS,

                 steps_per_epoch=int((nRow - TEST_SIZE) / BATCH),

                 validation_steps=int(TEST_SIZE / BATCH),

                 callbacks=[callback])
def pred(sentence):

    predicted = simple_model.predict(np.array([encode_str(sentence)], dtype=np.int64))

    print("{}-Prediction: {}".format(sentence, predicted))



pred("한남충 개돼지 뒤져라 좆팔")

pred("안녕하세요? 반갑습니다.")

pred("ㅎㄴㅊ 재기해")

pred("안녕하세요 ㅎㄴㅊ ㅅㅋ야")