import gc
import os
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as L
from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

SEED = 42
seed_everything(SEED)
develop_df = pd.read_csv('../input/lish-moa/train_features.csv')
develop_target_df = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
test_df = pd.read_csv('../input/lish-moa/test_features.csv')
sub = pd.read_csv('../input/lish-moa/sample_submission.csv')

target_cols = develop_target_df.columns[1:]
N_TARGETS = len(target_cols)
def preprocess_df(df):
    if 'sig_id' in df.columns:
        df = df.drop('sig_id', axis=1)
    df = df.drop('cp_type', axis=1)
    #df['cp_type'] = (df['cp_type'] == 'trt_cp').astype(int)
    df['cp_dose'] = (df['cp_dose'] == 'D2').astype(int)
    df['cp_time'] = df['cp_time'].map({24:0, 48: 1, 72: 2})
    return df
x_develop = preprocess_df(develop_df)
y_develop = develop_target_df.drop('sig_id', axis=1)
x_test = preprocess_df(test_df)
scaler = StandardScaler()

x_develop = pd.DataFrame(scaler.fit_transform(x_develop), columns=x_develop.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
def create_folds(df, fold_no, fold_type='mls_kfold', save=False):
    """
    df: target dataframe
    """
    if fold_type == 'kfold':
        kf = KFold(n_splits=fold_no, shuffle=True, random_state=SEED)
    elif fold_type == 'mls_kfold':
        kf = MultilabelStratifiedKFold(n_splits=fold_no, random_state=SEED)
        
    df['Fold'] = -1
    df.reset_index(inplace=True)
    for fold, (t, v) in enumerate(kf.split(df, df)):
        df.loc[v, 'Fold'] = fold
    df.set_index('sig_id', inplace=True)    
    if save:
        df.to_csv('Folds.csv')
    return df
class MyModel():
    def __init__(self, input_shape, N_TARGETS):
        self.input_shape = input_shape
        self.output_shape = N_TARGETS
        
    def create_model(self, output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        
        inputs = tf.keras.Input(shape=self.input_shape)
        x = L.BatchNormalization()(inputs)
        x = L.Dropout(0.5)(x)
        x = tfa.layers.WeightNormalization(L.Dense(100, kernel_initializer='he_normal'))(x)
        x = L.Activation('relu')(x)
        x = L.BatchNormalization()(x)
        x = L.Dropout(0.3)(x)
        x = tfa.layers.WeightNormalization(L.Dense(100, kernel_initializer='he_normal'))(x)
        x = L.Activation('relu')(x)
        x = L.BatchNormalization()(x)
        x = L.Dropout(0.2)(x)
        outputs = tfa.layers.WeightNormalization(L.Dense(self.output_shape,
                                                         activation='sigmoid',
                                                         bias_initializer=output_bias
                                                        ))(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        OPTIMIZER = tfa.optimizers.Lookahead(
            tfa.optimizers.AdamW(weight_decay=1e-4),
            sync_period=5)
        LOSS = tf.keras.losses.BinaryCrossentropy()
        
        model.compile(optimizer=OPTIMIZER, loss=LOSS)
        return model
def cv(output_bias):
    N_FOLDS = 5
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1984)

    models = {i: '' for i in range(N_FOLDS)}
    oof = {i: '' for i in range(N_FOLDS)}

    for foldno, (t_i, v_i) in enumerate(kf.split(x_develop, y_develop)):
        # Data Organization
        x_train = x_develop.loc[t_i]
        y_train = y_develop.loc[t_i]
        x_val = x_develop.loc[v_i]
        y_val = y_develop.loc[v_i]

        print(f"\nFold-%d" % (foldno))
        print("Train sample size:", x_train.shape[0], ", Validation sample size:", x_val.shape[0], '\n')

        # Training + Model Setup
        cb_es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        N_TARGET = y_develop.shape[1]
        input_shape = x_develop.shape[1]
        
        models[foldno] = MyModel(input_shape, N_TARGETS).create_model(output_bias=output_bias)

        # Training
        models[foldno].fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3, batch_size=48, callbacks=[cb_es])

        # Evaluation
        oof[foldno] = models[foldno].evaluate(x_val, y_val)
    
    # Out-of-fold Score
    mean_oof = np.mean(list(oof.values()))
    std_oof = np.std(list(oof.values()))
    
    print(f'\nOut-of-fold Score: %.5f +/- %.5f' % (mean_oof, std_oof))
        
    return models, oof
if len(x_test) == 3982:
    models, oof = cv(output_bias=0)
    submission = pd.DataFrame(data=0, columns=sub.columns[1:], index=sub['sig_id'])
    for i in models:
        submission += pd.DataFrame(data=models[i].predict(x_test), columns=sub.columns[1:], index=sub['sig_id'])
    submission = submission / len(models)
    submission.to_csv('submission.csv')
else:
    sub.to_csv('submission.csv', index=False)
if len(x_test) == 3982:
    models, oof = cv(output_bias=-np.log(y_develop.mean(axis=0).to_numpy()))
    submission = pd.DataFrame(data=0, columns=sub.columns[1:], index=sub['sig_id'])
    for i in models:
        submission += pd.DataFrame(data=models[i].predict(x_test), columns=sub.columns[1:], index=sub['sig_id'])
    submission = submission / len(models)
    submission.to_csv('submission.csv')
else:
    sub.to_csv('submission.csv', index=False)