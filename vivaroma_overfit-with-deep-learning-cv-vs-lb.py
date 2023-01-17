# reference: https://www.kaggle.com/braquino/neural-net-experimentation



from keras.models import Model

from keras.layers import *

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold 

from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss

import os

print(os.listdir("../input"))

df = pd.read_csv('../input/train.csv')

df.head()



# features selected by RFECV with lasso

features = ['16', '33', '43', '45', '52', '63', '65', '73', '90', '91', '117', '133', '134', '149', '189', '199', '217', '237', '258', '295']
X = df[features].values

y = df.values[:,1]

print(X.shape, y.shape)
def simple_model(input_shape):

    """

    define neural network model

    """

    inp = Input(shape=(input_shape[1],))

    x = Dense(3, activation='sigmoid')(inp)

    # only keep this layer, then the model becomes logistic regression

    x = Dense(1, activation='sigmoid')(x)

    

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.005), metrics=['accuracy'])

    return model
N_SPLITS = 10

splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True).split(X, y))

preds_val = []

y_val = []

best_models = []



for idx, (train_idx, val_idx) in enumerate(splits):

    print("Beginning fold {}".format(idx+1))

    X_train, y_train, X_val, y_val = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    model = simple_model(X_train.shape)

    cb = ModelCheckpoint('weights.h5', monitor='val_acc', mode='max', save_best_only=True, save_weights_only=True)

    model.fit(X_train, y_train, epochs=200, validation_data=(X_val, y_val), callbacks=[cb], verbose=0)

    model.load_weights('weights.h5')

    score = roc_auc_score(y_val, model.predict(X_val))

    print((model, score))

    best_models.append((model, score))
df_test = pd.read_csv('../input/test.csv')

print(len(df_test))

df_test.head()
X_test = df_test[features].values
y_preds = []

for mod, score in best_models:

    y_preds.append(mod.predict(X_test))

y_preds = np.concatenate(y_preds, axis=1)

y_preds.shape
subs = pd.read_csv('../input/sample_submission.csv')

mean_preds = y_preds.mean(axis=1)

subs['target'] = mean_preds

subs.to_csv('submission.csv', index=False)