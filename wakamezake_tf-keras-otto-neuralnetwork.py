import gc

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, ReLU, Input

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss
!ls ../input/otto-group-product-classification-challenge
train = pd.read_csv("../input/otto-group-product-classification-challenge/train.csv")

test = pd.read_csv("../input/otto-group-product-classification-challenge/test.csv")
def build_model(_input_shape, _num_classes):

    model = Sequential()

    model.add(Dense(512, input_dim=_input_shape, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(512, input_dim=_input_shape, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(512, input_dim=_input_shape, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    

    model.add(Dense(512, activation='relu'))

    model.add(Dense(_num_classes, activation="softmax"))

    return model
def preprocessing(_train, _test):

    drop_cols = ["id"]

    target_col = "target"

    feat_cols = [col for col in train.columns if col not in drop_cols + [target_col]]



    data = pd.concat([_train, _test]).reset_index()

    data_scale = StandardScaler().fit_transform(data[feat_cols])

    data = pd.concat([data[drop_cols + [target_col]], pd.DataFrame(data_scale)], axis=1)



    _train = data[~data[target_col].isnull()].reset_index(drop=True)

    _test = data[data[target_col].isnull()].reset_index(drop=True)



    _target = _train[target_col]

    _train.drop(columns=drop_cols + [target_col], inplace=True)

    _test.drop(columns=drop_cols + [target_col], inplace=True)



    _target = LabelEncoder().fit_transform(target)



    input_shapes = _train.shape[1]

    num_classes = np.unique(_target).size



    del data

    gc.collect()

    return _train, _test, _target
drop_cols = ["id"]

target_col = "target"

feat_cols = [col for col in train.columns if col not in drop_cols + [target_col]]



data = pd.concat([train, test]).reset_index()

data_scale = StandardScaler().fit_transform(data[feat_cols])

data = pd.concat([data[drop_cols + [target_col]], pd.DataFrame(data_scale)], axis=1)



train = data[~data[target_col].isnull()].reset_index(drop=True)

test = data[data[target_col].isnull()].reset_index(drop=True)



target = train[target_col]

train.drop(columns=drop_cols + [target_col], inplace=True)

test.drop(columns=drop_cols + [target_col], inplace=True)



target = LabelEncoder().fit_transform(target)



input_shapes = train.shape[1]

num_classes = np.unique(target).size



del data

gc.collect()
train.shape, test.shape
input_shapes, num_classes
np.unique(target).size
EPOCHS = 20

BATCH_SIZE = 512

NFOLDS = 5

RANDOM_STATE = 871972



folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, 

                        random_state=RANDOM_STATE)
y_pred_test = np.zeros((len(test), 9))

oof = np.zeros((len(train), 9))

score = 0



for fold_n, (train_index, valid_index) in enumerate(folds.split(train, y=target)):

    print('Fold', fold_n)

    X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]

    y_train, y_valid = target[train_index], target[valid_index]

    model = build_model(input_shapes, num_classes)

    model.compile(optimizer='adam',

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])

    history = model.fit(train, target, epochs=EPOCHS, batch_size=BATCH_SIZE)

    y_pred_valid = model.predict(X_valid)

    oof[valid_index] = y_pred_valid

    score += log_loss(y_valid, y_pred_valid)

    y_pred_test += model.predict(test) / NFOLDS

print('valid logloss average:', score / NFOLDS, log_loss(target, oof))
sample_submit = pd.read_csv("../input/otto-group-product-classification-challenge/sampleSubmission.csv")

submit = pd.concat([sample_submit[['id']], pd.DataFrame(y_pred_test)], axis = 1)

submit.columns = sample_submit.columns

submit.to_csv('submit.csv', index=False)
np.save("keras_oof.npy", oof)