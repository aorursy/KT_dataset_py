
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
import warnings

LEN_WORDS = 30
LEN_EMBEDDING = 300
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
X_train = train["question_text"].fillna("fillna").values
y_train = train["target"].values
X_test = test["question_text"].fillna("fillna").values

max_features = 30000
maxlen = 40
embed_size = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
class F1Evaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred = (y_pred > 0.5).astype(int)
            score = f1_score(self.y_val, y_pred)
            print("\n F1 Score - epoch: %d - score: %.6f \n" % (epoch+1, score))
from keras.models import Model
from keras.layers import Conv1D, Input, MaxPooling1D, Flatten, Dense, BatchNormalization, concatenate
from keras.layers import LeakyReLU, Activation
STRIDE_1 = 2
STRIDE_2 = 4
STRIDE_3 = 8

FILTER_1 = 64
FILTER_2 = 64
FILTER_3 = 64

inp = Input(shape=(maxlen, ))
embed_layer = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
embed_layer = SpatialDropout1D(0.4)(embed_layer)
embed_layer = Reshape((maxlen, embed_size))(embed_layer)

# line1 = BatchNormalization()(Input_layer)
line1 = Conv1D(FILTER_1, STRIDE_1)(embed_layer)
line1 = Activation(LeakyReLU())(line1)
line1 = MaxPooling1D(STRIDE_1)(line1)
line1 = Conv1D(FILTER_1, STRIDE_1)(line1)
line1 = Activation(LeakyReLU())(line1)
line1 = MaxPooling1D(STRIDE_1)(line1)
line1 = Conv1D(FILTER_1, STRIDE_1)(line1)
line1 = Activation(LeakyReLU())(line1)
line1 = MaxPooling1D(STRIDE_1*2)(line1)  # global max pooling
line1 = Flatten()(line1)

# line2 = BatchNormalization()(Input_layer)
line2 = Conv1D(FILTER_2, STRIDE_1)(embed_layer)
line2 = Activation(LeakyReLU())(line2)
line2 = MaxPooling1D(STRIDE_1)(line2)
line2 = Conv1D(FILTER_2, STRIDE_1)(line2)
line2 = Activation(LeakyReLU())(line2)
line2 = MaxPooling1D(STRIDE_1)(line2)
line2 = Conv1D(FILTER_2, STRIDE_1)(line2)
line2 = Activation(LeakyReLU())(line2)
line2 = MaxPooling1D(STRIDE_1*2)(line2)  # global max pooling
line2 = Flatten()(line2)

# line3 = BatchNormalization()(Input_layer)
line3 = Conv1D(FILTER_3, STRIDE_1)(embed_layer)
line3 = Activation(LeakyReLU())(line3)
line3 = MaxPooling1D(STRIDE_1)(line3)
line3 = Conv1D(FILTER_3, STRIDE_1)(line3)
line3 = Activation(LeakyReLU())(line3)
line3 = MaxPooling1D(STRIDE_1)(line3)
line3 = Conv1D(FILTER_3, STRIDE_1)(line3)
line3 = Activation(LeakyReLU())(line3)
line3 = MaxPooling1D(STRIDE_1*2)(line3)  # global max pooling
line3 = Flatten()(line3)

concat_layer = concatenate([line1, line2, line3])

total = Dense(1024, activation='relu')(concat_layer)
preds = Dense(1, activation='sigmoid')(total)

model = Model(inputs=inp, outputs=preds)
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
batch_size = 256
epochs = 10

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95,
                                              random_state=1989)
F1_Score = F1Evaluation(validation_data=(X_val, y_val), interval=1)

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs,
                 validation_data=(X_val, y_val),
                 callbacks=[F1_Score], verbose=1)
y_pred = model.predict(x_test, batch_size=1024)
y_pred = (y_pred > 0.5).astype(int)
submission['prediction'] = y_pred
submission.to_csv('submission.csv', index=False)