import pandas as pd

df = pd.read_csv("../input/clinical_trial.csv")

print("no abstract : ", sum(df['abstract'].isnull()))

print("no title : ", sum(df['title'].isnull()))



df['abstract'] = df['abstract'].fillna(value="")
from sklearn import *

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np



titles = df['title']

abstracts = df['abstract']

y = np.array(df['trial'])



pre_vect_title = TfidfVectorizer(stop_words = 'english')

pre_vect_abst = TfidfVectorizer(stop_words = 'english')



title_vectors = pre_vect_title.fit_transform(titles)

abst_vectors = pre_vect_abst.fit_transform(abstracts)



from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(100)

title_vectors = svd.fit_transform(title_vectors)

abst_vectors = svd.fit_transform(abst_vectors)



# check the type of the data

print(title_vectors.shape)

print(abst_vectors.shape)



all_data = np.concatenate((title_vectors, abst_vectors), axis = 1)

print(all_data.shape)



def history_plot(history):

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()



    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.optimizers import SGD

from keras.utils.vis_utils import plot_model



# define model

model = Sequential()

model.add(Dense(input_dim=100, units=1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.10), metrics=['accuracy'])



history = model.fit(title_vectors, y, epochs=800, batch_size=10, validation_split=0.25, verbose=0)
import matplotlib.pyplot as plt

history_plot(history)
history2 = model.fit(abst_vectors, y, epochs=1000, batch_size=10, validation_split=0.25, verbose=0)
history_plot(history2)
model_comb = Sequential()

model_comb.add(Dense(input_dim=200, units=1))

model_comb.add(Activation('sigmoid'))

model_comb.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.10), metrics=['accuracy'])



history_comb = model_comb.fit(all_data, y, epochs=1000, batch_size=10, validation_split=0.25, verbose=0)
history_plot(history_comb)
from keras.layers import Dropout



# define multi-layer model

model_mul = Sequential()

model_mul.add(Dense(200, input_dim=200, init='normal', activation='relu'))

model_mul.add(Dropout(0.5))

model_mul.add(Dense(1, init='normal', activation='sigmoid'))

model_mul.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



history_mul = model_mul.fit(all_data, y, epochs=200, batch_size=10, validation_split=0.50, verbose=0)
history_plot(history_mul)
# define multi-layer model

model_mul2 = Sequential()

model_mul2.add(Dense(40, input_dim=200, init='normal', activation='relu'))

model_mul2.add(Dropout(0.8))

model_mul2.add(Dense(1, init='normal', activation='sigmoid'))

model_mul2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



history_mul2 = model_mul2.fit(all_data, y, epochs=200, batch_size=10, validation_split=0.50, verbose=0)
history_plot(history_mul2)
print(max(history.history['val_acc']))

print(max(history2.history['val_acc']))

print(max(history_comb.history['val_acc']))

print(max(history_mul.history['val_acc']))