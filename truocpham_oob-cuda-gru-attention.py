import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
from tqdm import tqdm

print(os.listdir("../input"))
from keras import Sequential
from keras import optimizers
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model
from keras.layers import LSTM, Dense, Bidirectional, Input,Dropout,BatchNormalization,CuDNNLSTM, GRU, CuDNNGRU, Embedding, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import roc_curve, auc
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
sample_submission = pd.read_csv('../input/sample_submission.csv')
train.shape, test.shape, sample_submission.shape
train.head()
# https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
def get_model():
    model = Sequential()
    model.add(BatchNormalization(momentum=0.98, input_shape=(10, 128)))
    model.add(Bidirectional(CuDNNGRU(128, return_sequences=True)))
    model.add(Attention(10))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
    return model
# Make data for training
xtrain = [k for k in train['audio_embedding']]
test_data = test['audio_embedding'].tolist()
ytrain = train['is_turkey'].values

# Pad the audio features so that all are "10 seconds" long
x_train = pad_sequences(xtrain, maxlen=10)
y_train = np.asarray(ytrain)
kf = KFold(n_splits=10, shuffle=True, random_state=42069)
test_data = pad_sequences(test_data)
oof_preds = []
aucs = 0

for n_fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
    x_train_f = x_train[train_idx]
    y_train_f = y_train[train_idx]
    x_val_f = x_train[val_idx]
    y_val_f = y_train[val_idx]
    
    # Get model
    model = get_model()
    
    # Fit
    model.fit(x_train_f, y_train_f,
              batch_size=256,
              epochs=12,
              verbose=0,
              validation_data=(x_val_f, y_val_f))

    # Get accuracy of model on validation data. It's not AUC but it's something at least!
    preds_val = model.predict([x_val_f], batch_size=512)
    oof_preds.append(model.predict(test_data))

    fpr, tpr, thresholds = roc_curve(y_val_f, preds_val, pos_label=1)
    aucs += auc(fpr,tpr)
    print('Fold {}, AUC = {}'.format(n_fold, auc(fpr, tpr)))

print("Cross Validation AUC = {}".format(aucs/10))
preds = np.asarray(oof_preds)[...,0]
preds = np.mean(preds, axis=0)
submission = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':preds})
print(submission.head(5))
def get_pseudo_data(sub_df, x_train, y_train, pos_threshold=0.99, neg_threshold=0.01):
    pred_probs = sub_df.is_turkey.values
    pseudo_index = np.argwhere(np.logical_or(pred_probs > pos_threshold, pred_probs < neg_threshold ))[:,0]
    
    pseudo_x_train = test_data[pseudo_index]
    pseudo_y_train = pred_probs[pseudo_index]
    pseudo_y_train[pseudo_y_train > 0.5] = 1
    pseudo_y_train[pseudo_y_train <= 0.5] = 0
    
    X = np.concatenate([x_train, pseudo_x_train], axis=0)
    y = np.concatenate([y_train, pseudo_y_train])
    
    return X, y
x_train, y_train = get_pseudo_data(submission, x_train, y_train)
kf = KFold(n_splits=10, shuffle=True, random_state=42069)
test_data = pad_sequences(test_data)
oof_preds = []
aucs = 0

for n_fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
    x_train_f = x_train[train_idx]
    y_train_f = y_train[train_idx]
    x_val_f = x_train[val_idx]
    y_val_f = y_train[val_idx]
    
    # Get model
    model = get_model()
    
    # Fit
    model.fit(x_train_f, y_train_f,
              batch_size=256,
              epochs=12,
              verbose=0,
              validation_data=(x_val_f, y_val_f))

    # Get accuracy of model on validation data. It's not AUC but it's something at least!
    preds_val = model.predict([x_val_f], batch_size=512)
    oof_preds.append(model.predict(test_data))

    fpr, tpr, thresholds = roc_curve(y_val_f, preds_val, pos_label=1)
    aucs += auc(fpr,tpr)
    print('Fold {}, AUC = {}'.format(n_fold, auc(fpr, tpr)))

print("Cross Validation AUC = {}".format(aucs/10))
# Get submisison 1 using threshold
preds = np.asarray(oof_preds)[...,0]
preds = np.mean(preds, axis=0)
sub_df = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':preds})
sub_df.to_csv('submission1.csv', index=False)
print(sub_df.head(10))
print(sub_df.shape)
n_bags = 20
n_folds = 10
random_state = 0
bag_oof_preds = []
aucs = 0

for n_bag in range(n_bags):

    random_state += n_bag
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    test_data = pad_sequences(test_data)
    oof_preds = []
    
    print('--> OOB #{}'.format(n_bag))

    # Out-of-Fold Method
    for n_fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
        x_train_f = x_train[train_idx]
        y_train_f = y_train[train_idx]
        x_val_f = x_train[val_idx]
        y_val_f = y_train[val_idx]

        # Get model
        model = get_model()

        # Fit
        model.fit(x_train_f, y_train_f,
                  batch_size=256,
                  epochs=12,
                  verbose=0,
                  validation_data=(x_val_f, y_val_f))

        # Get accuracy of model on validation data. It's not AUC but it's something at least!
        preds_val = model.predict([x_val_f], batch_size=512)
        oof_preds.append(model.predict(test_data))

        fpr, tpr, thresholds = roc_curve(y_val_f, preds_val, pos_label=1)
        aucs += auc(fpr,tpr)
        print('Fold {}, AUC = {}'.format(n_fold, auc(fpr, tpr)))

    bag_oof_preds.append(oof_preds)

print("Full Cross Validation AUC = {}".format(aucs/(n_bags*n_folds)))
oob_preds = np.asarray(bag_oof_preds)[...,0]
mean_preds = np.mean(oob_preds, axis=0)
preds = np.mean(mean_preds, axis=0)
submission = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':preds})
submission.to_csv('submission2.csv', index=False)
print(submission.head(5))
print(submission.shape)