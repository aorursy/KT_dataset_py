import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
import warnings

from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional, Dropout
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
## split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
        return 
 
my_metrics = Metrics()
from keras.models import Model
from keras.layers import Conv1D, Input, MaxPooling1D, Flatten, Dense, BatchNormalization, concatenate, SpatialDropout1D
from keras.layers.advanced_activations import LeakyReLU

# class LeakyReLU(LeakyReLU):
#     def __init__(self, **kwargs):
#         self.__name__ = "LeakyReLU"
#         super(LeakyReLU, self).__init__(**kwargs)
def f1_loss(y_true, y_pred):
    """Custom f1 loss for bicategorical
    y must be of shape where y.shape[-1] == 2
    y[..., 0] must be the category for true
    y[..., 1] must be the category for false
    """
    true_truth = K.dot(y_true, K.constant([1., 0.], dtype='float32', shape=(2, 1)))
    true_false = K.dot(y_true, K.constant([0., 1.], dtype='float32', shape=(2, 1)))

    y_false = K.constant(1., dtype='float32') - y_true

    fake_truth = K.dot(y_false, K.constant([1., 0.], dtype='float32', shape=(2, 1)))
    fake_false = K.dot(y_false, K.constant([0., 1.], dtype='float32', shape=(2, 1)))

    TP_temp = K.sum(true_truth * y_pred)
    FP_temp = K.sum(fake_truth * y_pred)
    FN_temp = K.sum(fake_false * y_pred)

    loss = (FP_temp + FN_temp) / (2 * TP_temp + FP_temp + FN_temp + K.epsilon())
!ls ../input/embeddings/
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
############### CNN 1D

STRIDE_1 = 2
STRIDE_2 = 4
STRIDE_3 = 8

FILTER_1 = 64
FILTER_2 = 64
FILTER_3 = 64

inp = Input(shape=(maxlen, ))
embed_layer1 = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
embed_layer1 = SpatialDropout1D(0.4)(embed_layer1)

# line1 = BatchNormalization()(Input_layer)
line1 = Conv1D(FILTER_1, STRIDE_1)(embed_layer1)
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
line2 = Conv1D(FILTER_2, STRIDE_1)(embed_layer1)
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
line3 = Conv1D(FILTER_3, STRIDE_1)(embed_layer1)
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

conv1d_dense = Dense(1024, activation='relu')(concat_layer)



############### RNN
embed_layer2 = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
embed_layer2 = SpatialDropout1D(0.4)(embed_layer2)

rnn_line = Bidirectional(CuDNNLSTM(64, return_sequences=True), input_shape=(maxlen, embed_size))(embed_layer2)
rnn_line = Bidirectional(CuDNNLSTM(64))(rnn_line)
rnn_dense = Dense(1024, activation='relu')(rnn_line)



############### CNN 2D

filter_sizes = [1,2,3,5]
num_filters = 36

embed_layer3 = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
embed_layer3 = SpatialDropout1D(0.4)(embed_layer3)

x = SpatialDropout1D(0.4)(embed_layer3)
x = Reshape((maxlen, embed_size, 1))(x)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size),
                             kernel_initializer='he_normal', activation='elu')(x)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size),
                             kernel_initializer='he_normal', activation='elu')(x)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), 
                             kernel_initializer='he_normal', activation='elu')(x)
conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size),
                             kernel_initializer='he_normal', activation='elu')(x)

maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)

cnn2d_dense = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
cnn2d_dense = Flatten()(cnn2d_dense)
cnn2d_dense = Dropout(0.1)(cnn2d_dense)

total = concatenate([cnn2d_dense, conv1d_dense, rnn_dense])

preds = Dense(1, activation='sigmoid')(total)

model = Model(inputs=inp, outputs=preds)
from keras.utils import plot_model
plot_model(model, to_file='model.png')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', f1])
## Train the model 
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_glove_val_y = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_glove_val_y>thresh).astype(int))))
pred_glove_test_y = model.predict([test_X], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x
import gc; gc.collect()
time.sleep(10)
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
############### CNN 1D

STRIDE_1 = 2
STRIDE_2 = 4
STRIDE_3 = 8

FILTER_1 = 64
FILTER_2 = 64
FILTER_3 = 64

inp = Input(shape=(maxlen, ))
embed_layer1 = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
embed_layer1 = SpatialDropout1D(0.4)(embed_layer1)

# line1 = BatchNormalization()(Input_layer)
line1 = Conv1D(FILTER_1, STRIDE_1)(embed_layer1)
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
line2 = Conv1D(FILTER_2, STRIDE_1)(embed_layer1)
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
line3 = Conv1D(FILTER_3, STRIDE_1)(embed_layer1)
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

conv1d_dense = Dense(1024, activation='relu')(concat_layer)



############### RNN
embed_layer2 = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
embed_layer2 = SpatialDropout1D(0.4)(embed_layer2)

rnn_line = Bidirectional(CuDNNLSTM(64, return_sequences=True), input_shape=(maxlen, embed_size))(embed_layer2)
rnn_line = Bidirectional(CuDNNLSTM(64))(rnn_line)
rnn_dense = Dense(1024, activation='relu')(rnn_line)



############### CNN 2D

filter_sizes = [1,2,3,5]
num_filters = 36

embed_layer3 = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
embed_layer3 = SpatialDropout1D(0.4)(embed_layer3)

x = SpatialDropout1D(0.4)(embed_layer3)
x = Reshape((maxlen, embed_size, 1))(x)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size),
                             kernel_initializer='he_normal', activation='elu')(x)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size),
                             kernel_initializer='he_normal', activation='elu')(x)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), 
                             kernel_initializer='he_normal', activation='elu')(x)
conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size),
                             kernel_initializer='he_normal', activation='elu')(x)

maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)

cnn2d_dense = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
cnn2d_dense = Flatten()(cnn2d_dense)
cnn2d_dense = Dropout(0.1)(cnn2d_dense)

total = concatenate([cnn2d_dense, conv1d_dense, rnn_dense])

preds = Dense(1, activation='sigmoid')(total)

model = Model(inputs=inp, outputs=preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', f1])
## Train the model 
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_fasttext_val_y = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_fasttext_val_y>thresh).astype(int))))
pred_fasttext_test_y = model.predict([test_X], batch_size=1024, verbose=1)
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x
import gc; gc.collect()
time.sleep(10)
EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
############### CNN 1D

STRIDE_1 = 2
STRIDE_2 = 4
STRIDE_3 = 8

FILTER_1 = 64
FILTER_2 = 64
FILTER_3 = 64

inp = Input(shape=(maxlen, ))
embed_layer1 = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
embed_layer1 = SpatialDropout1D(0.4)(embed_layer1)

# line1 = BatchNormalization()(Input_layer)
line1 = Conv1D(FILTER_1, STRIDE_1)(embed_layer1)
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
line2 = Conv1D(FILTER_2, STRIDE_1)(embed_layer1)
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
line3 = Conv1D(FILTER_3, STRIDE_1)(embed_layer1)
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

conv1d_dense = Dense(1024, activation='relu')(concat_layer)



############### RNN
embed_layer2 = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
embed_layer2 = SpatialDropout1D(0.4)(embed_layer2)

rnn_line = Bidirectional(CuDNNLSTM(64, return_sequences=True), input_shape=(maxlen, embed_size))(embed_layer2)
rnn_line = Bidirectional(CuDNNLSTM(64))(rnn_line)
rnn_dense = Dense(1024, activation='relu')(rnn_line)



############### CNN 2D

filter_sizes = [1,2,3,5]
num_filters = 36

embed_layer3 = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
embed_layer3 = SpatialDropout1D(0.4)(embed_layer3)

x = SpatialDropout1D(0.4)(embed_layer3)
x = Reshape((maxlen, embed_size, 1))(x)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size),
                             kernel_initializer='he_normal', activation='elu')(x)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size),
                             kernel_initializer='he_normal', activation='elu')(x)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), 
                             kernel_initializer='he_normal', activation='elu')(x)
conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size),
                             kernel_initializer='he_normal', activation='elu')(x)

maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)

cnn2d_dense = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
cnn2d_dense = Flatten()(cnn2d_dense)
cnn2d_dense = Dropout(0.1)(cnn2d_dense)

total = concatenate([cnn2d_dense, conv1d_dense, rnn_dense])

preds = Dense(1, activation='sigmoid')(total)

model = Model(inputs=inp, outputs=preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', f1])
## Train the model 
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_paragram_val_y = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_paragram_val_y>thresh).astype(int))))
pred_paragram_test_y = model.predict([test_X], batch_size=1024, verbose=1)
pred_val_y = 0.33*pred_glove_val_y + 0.33*pred_fasttext_val_y + 0.34*pred_paragram_val_y 
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))))
pred_test_y = 0.33*pred_glove_test_y + 0.33*pred_fasttext_test_y + 0.34*pred_paragram_test_y
pred_test_y = (pred_test_y>0.35).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
