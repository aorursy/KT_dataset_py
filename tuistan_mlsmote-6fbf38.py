!pip install ../input/iterative-stratification/iterative-stratification-master
import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

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
x_develop = pd.read_csv('../input/lish-moa/train_features.csv')
y_develop = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
x_test = pd.read_csv('../input/lish-moa/test_features.csv')
sub = pd.read_csv('../input/lish-moa/sample_submission.csv')

target_cols = y_develop.columns[1:]
N_TARGETS = len(target_cols)
def preprocess_df(df):
    if 'cp_type' in df.columns:
        df = df.drop('cp_type', axis=1)
    df['cp_dose'] = (df['cp_dose'] == 'D2').astype(int)
    df['cp_time'] = df['cp_time'].map({24:1, 48: 2, 72: 3})
    return df
x_develop = preprocess_df(x_develop)
x_test = preprocess_df(x_test)

x_develop = x_develop.drop('sig_id', axis=1)
y_develop = y_develop.drop('sig_id', axis=1)
x_test = x_test.set_index('sig_id')
def create_folds(df, fold_no, fold_type='mls_kfold', save=False, seed=42):
    """
    df: target dataframe
    """
    if fold_type == 'kfold':
        kf = KFold(n_splits=fold_no, shuffle=True, random_state=seed)
    elif fold_type == 'mls_kfold':
        kf = MultilabelStratifiedKFold(n_splits=fold_no, shuffle=False, random_state=seed)
        
    df['Fold'] = -1
    df.reset_index(inplace=True, drop=True)
    for fold, (t, v) in enumerate(kf.split(df, df)):
        df.loc[v, 'Fold'] = fold
    if save:
        df.to_csv('Folds.csv')
    return df
N_FOLDS = 5
fold_type = 'mls_kfold'
y_develop = create_folds(y_develop, fold_no=N_FOLDS, fold_type=fold_type, seed=SEED)
def create_model(input_shape, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    inputs = tf.keras.Input(shape=input_shape)
    x = L.BatchNormalization()(inputs)
    x = tfa.layers.WeightNormalization(L.Dense(1000, activation='swish'))(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.4)(x)
    x = tfa.layers.WeightNormalization(L.Dense(500, activation='swish'))(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.4)(x)
    outputs = tfa.layers.WeightNormalization(L.Dense(N_TARGETS,
                                                     activation='sigmoid',
                                                     bias_initializer=output_bias
                                                    )
                                            )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
# Taken from https://github.com/niteshsukhwani/MLSMOTE, modified by Tolga Dincer.
def get_tail_label(df: pd.DataFrame, ql=[0.05, 1.]) -> list:
    """
    Find the underrepresented targets.
    Underrepresented targets are those which are observed less than the median occurance.
    Targets beyond a quantile limit are filtered.
    """
    irlbl = df.sum(axis=0)
    irlbl = irlbl[(irlbl > irlbl.quantile(ql[0])) & ((irlbl < irlbl.quantile(ql[1])))]  # Filtering
    irlbl = irlbl.max() / irlbl
    threshold_irlbl = irlbl.median()
    tail_label = irlbl[irlbl > threshold_irlbl].index.tolist()
    return tail_label

def get_minority_samples(X: pd.DataFrame, y: pd.DataFrame, ql=[0.05, 1.]):
    """
    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    tail_labels = get_tail_label(y, ql=ql)
    index = y[y[tail_labels].apply(lambda x: (x == 1).any(), axis=1)].index.tolist()
    
    X_sub = X[X.index.isin(index)].reset_index(drop = True)
    y_sub = y[y.index.isin(index)].reset_index(drop = True)
    return X_sub, y_sub

def nearest_neighbour(X: pd.DataFrame, neigh) -> list:
    """
    Give index of 10 nearest neighbor of all the instance
    
    args
    X: np.array, array whose nearest neighbor has to find
    
    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(n_neighbors=neigh, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices

def MLSMOTE(X, y, n_sample, neigh=5):
    """
    Give the augmented data using MLSMOTE algorithm
    
    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample
    
    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X, neigh=5)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n-1)
        neighbor = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis = 0, skipna = True)
        target[i] = np.array([1 if val > 0 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference,:] - X.loc[neighbor,:]
        new_X[i] = np.array(X.loc[reference,:] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    return new_X, target

def SMOLTE_cat_wrapper(x_df, y_df, cat_col, nsamples):
    x_df_up = pd.DataFrame(columns=x_df.columns)
    y_df_up = pd.DataFrame(columns=y_df.columns)

    unique_cat_combs = x_df.groupby(cat_col).size().reset_index().rename(columns={0:'count'})[cat_col]
    num_cols = x_df.columns.drop(cat_col).tolist()
    for index, row in unique_cat_combs.iterrows():
        condition = (x_df[cat_col] == row).all(axis=1)

        subx = x_df[condition][num_cols].reset_index(drop=True)
        suby = y_df[condition].reset_index(drop=True)

        x_df_sub, y_df_sub = get_minority_samples(subx, suby)
        a, b = MLSMOTE(x_df_sub, y_df_sub, nsamples, 5)
        cats = pd.concat([row.to_frame().T]*len(a), ignore_index=True)
        a = pd.merge(cats, a, how='left', left_index=True, right_index=True)
        x_df_up = x_df_up.append(a, ignore_index=True)
        y_df_up = y_df_up.append(b, ignore_index=True)
    #y_df_up = y_df_up.astype(int)
    
    print('Number of new samples created: %d' %(len(y_df_up)))
    
    x_df_up = pd.concat([x_df, x_df_up], ignore_index=True)
    y_df_up = pd.concat([y_df, y_df_up], ignore_index=True)
    
    x_df_up = x_df_up.sample(len(x_df_up), random_state=1881).reset_index(drop=True)
    y_df_up = y_df_up.sample(len(y_df_up), random_state=1881).reset_index(drop=True)
    
    x_df_up[cat_col] = x_df_up[cat_col].astype(int)
    return x_df_up, y_df_up
def oof_score(oof: dict):
    return np.mean(list(oof.values())), np.std(list(oof.values()))


def run_cv(summary=True, debug=False, nsamples=100):
    histories = {x: '' for x in range(N_FOLDS)} 
    models = {x: '' for x in range(N_FOLDS)}
    results = {x: '' for x in range(N_FOLDS)}
    oof_bp = {x: [] for x in range(N_FOLDS)}
    oof_ap = {x: [] for x in range(N_FOLDS)}
    
    for foldno in np.sort(y_develop.Fold.unique()):
        x_train_fold = x_develop[y_develop.Fold != foldno]
        y_train_fold = y_develop[y_develop.Fold != foldno].drop('Fold', axis=1)
        x_val_fold = x_develop[y_develop.Fold == foldno]
        y_val_fold = y_develop[y_develop.Fold == foldno].drop('Fold', axis=1)
        
        train_sample_size = len(y_train_fold)
        val_sample_size = len(y_val_fold)
        print(" ")
        print(f"Fold-%d" % (foldno))
        print("Original Train sample size:", train_sample_size, ", Original validation sample size:", val_sample_size)

        FEATURE_SIZE = x_train_fold.shape[-1]
        
        cat_col = ['cp_time', 'cp_dose']
        x_train_fold, y_train_fold = SMOLTE_cat_wrapper(x_train_fold, y_train_fold, cat_col, nsamples=nsamples)
        print("Upsampled Train sample size: %d" % (len(x_train_fold)))
        
        # Train Data Pipeline
        train_ds = tf.data.Dataset.from_tensor_slices((x_train_fold, y_train_fold))
        train_ds = train_ds.shuffle(1024).batch(56)

        # Validation Data Pipeline
        val_ds = tf.data.Dataset.from_tensor_slices((x_val_fold, y_val_fold))
        val_ds = val_ds.batch(val_sample_size)

        # Callbacks
        cb_es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')

        # Model & Fit
        models[foldno] = create_model(FEATURE_SIZE, output_bias=8.32)
        OPTIMIZER = tfa.optimizers.Lookahead(
            tfa.optimizers.AdamW(weight_decay=1e-5),
            sync_period=5)
        models[foldno].compile(optimizer=OPTIMIZER, loss=tf.keras.losses.BinaryCrossentropy()) #label_smoothing=0.001
        histories[foldno] = models[foldno].fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[cb_es, reduce_lr_loss], verbose=1)
        
        # OOF (Validation) Results
        oof_y_val = models[foldno].predict(x_val_fold)
        
        oof_bp[foldno] = tf.keras.losses.binary_crossentropy(y_val_fold, oof_y_val).numpy().mean()
        print('Out-of-Fold Score: ', oof_bp[foldno])
        
        #oof_y_val[cp_type_val_fold == 0] = 0
        oof_ap[foldno] = tf.keras.losses.binary_crossentropy(y_val_fold, oof_y_val).numpy().mean()
        print('Out-of-Fold Score with post processing: ', oof_ap[foldno])
        
            
        # Test Predictions
        results[foldno] = pd.DataFrame(index=x_test.index,
                                       columns=target_cols,
                                       data=models[foldno].predict(x_test))
        #results[foldno].loc[cp_type_te.index[cp_type_te == 'ctl_vehicle']] = 0

        
        if foldno == 0:
            res = results[foldno]
        else:
            res += results[foldno]

        # Save Model
        if SAVE_MODEL:
            models[foldno].save(f'weights-fold{foldno}.h5')
    
    res = res / N_FOLDS
    print('\n')
    if summary:
        print('Summary')
        # Mean out of score before postprocessing
        print('Mean OOF score: %f +/- %f' % (oof_score(oof_bp)))

        # Mean out of score after postprocessing
        print('Mean OOF score after postprocessing: %f +/- %f' % (oof_score(oof_ap)))
    
    return res, histories, oof_ap
EPOCHS = 45
SAVE_MODEL = False

if sub.shape[0] != 3981:
    res, histories, oof_ap = run_cv(summary=True, debug=False, nsamples=50)
    sub = res.reset_index()
    sub.to_csv('submission.csv', index=False)
else:
    sub.to_csv('submission.csv', index=False)
sub.shape[0]
sub