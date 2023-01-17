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
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Bidirectional, Dropout
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
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
from keras.layers import Conv1D, Input, MaxPooling1D, Flatten, Dense, BatchNormalization, concatenate
from keras.layers import LeakyReLU, Activation
from keras.layers import LeakyReLU

class LeakyReLU(LeakyReLU):
    def __init__(self, **kwargs):
        self.__name__ = "LeakyReLU"
        super(LeakyReLU, self).__init__(**kwargs)

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
embed_layer2 = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
embed_layer2 = SpatialDropout1D(0.4)(embed_layer2)
rnn_line = Bidirectional(CuDNNLSTM(32, return_sequences=True), input_shape=(maxlen, embed_size))(embed_layer2)
rnn_line = Bidirectional(CuDNNLSTM(32))(rnn_line)
rnn_dense = Dense(1024, activation='relu')(rnn_line)

total = concatenate([conv1d_dense, rnn_dense])
preds = Dense(1, activation='sigmoid')(total)

model = Model(inputs=inp, outputs=preds)
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model.png')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
from keras import backend as K
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

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', f1])
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95,
                                              random_state=1989)
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
my_weights = '../working/2_channel.h5'
try:
    model.load_weights(my_weights)
    print('Load weights')
except:
    pass
batch_size = 256
epochs = 50

from keras.callbacks import EarlyStopping, ModelCheckpoint
check_point = ModelCheckpoint(my_weights, monitor="val_f1", mode="max",
                              verbose=True, save_best_only=True)
early_stop = EarlyStopping(monitor="val_f1", mode="max", patience=8,verbose=True)
F1_Score = F1Evaluation(validation_data=(X_val, y_val), interval=1)

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs,
                 validation_data=(X_val, y_val),
                 callbacks=[early_stop, check_point, F1_Score], verbose=1)
from sklearn import metrics

pred_noemb_val_y = model.predict([X_val], batch_size=1024, verbose=1)
scores_list = dict()
for thresh in np.arange(0.1, 0.6, 0.01):
    thresh = np.round(thresh, 2)
    temp_score = metrics.f1_score(y_val, (pred_noemb_val_y>thresh).astype(int))
    scores_list[thresh] = temp_score
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_val, (pred_noemb_val_y>thresh).astype(int))))
opt_threshold = max(scores_list, key=scores_list.get)
y_pred = model.predict(x_test, batch_size=1024, verbose=1)
y_pred = (y_pred > opt_threshold).astype(int)
submission['prediction'] = y_pred
submission.to_csv('submission.csv', index=False)







