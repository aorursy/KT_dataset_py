# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 
import zipfile
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, scipy, sklearn, xgboost as xgb
import IPython
import copy
import warnings
import re
#
import tensorflow as tf
from tensorflow.keras import layers,optimizers,Model
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
import pickle
#
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.multioutput import MultiOutputRegressor,MultiOutputClassifier
from sklearn.metrics import log_loss, accuracy_score
#
%matplotlib inline

TRAIN_FEATURES_PATH = "/kaggle/input/lish-moa/train_features.csv"
TEST_FEATURES_PATH = "/kaggle/input/lish-moa/test_features.csv"
TRAIN_TARGETS_NONSCORED_PATH = "/kaggle/input/lish-moa/train_targets_nonscored.csv"
SAMPLE_SUB_PATH = "/kaggle/input/lish-moa/sample_submission.csv"
TRAIN_TARGET_SCORE_PATH = "/kaggle/input/lish-moa/train_targets_scored.csv"
###
pdf01A_trfeat = pd.read_csv(TRAIN_FEATURES_PATH).sort_values(by="sig_id")
pdf01B_trtar = pd.read_csv(TRAIN_TARGET_SCORE_PATH).sort_values(by="sig_id")
#
pdf02A_tefeat = pd.read_csv(TEST_FEATURES_PATH).sort_values(by="sig_id")
pdf02B_tetar = pd.read_csv(SAMPLE_SUB_PATH).sort_values(by="sig_id")
#
pdf03_train_nonscored = pd.read_csv(TRAIN_TARGETS_NONSCORED_PATH).sort_values(by="sig_id")
###
target_cols = [i_cname for i_cname in pdf01B_trtar.columns 
    if i_cname not in ["sig_id"]]
feat_cols = [i_cname for i_cname in pdf01A_trfeat.columns
    if i_cname not in ["sig_id"]]
###
def encoder(pdf01A_trfeat:pd.DataFrame, 
        pdf02A_tefeat:pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    print("encode cp_type and cp_dose")
    #
    from sklearn.preprocessing import LabelEncoder
    #
    for feature in ["cp_type","cp_dose","cp_time"]:
        trans = LabelEncoder()
        trans.fit(list(pdf01A_trfeat[feature].astype(str).values) +
                  list(pdf02A_tefeat[feature].astype(str).values))
        pdf01A_trfeat[feature] = trans.transform(list(pdf01A_trfeat[feature].astype(str).values))
        pdf02A_tefeat[feature] = trans.transform(list(pdf02A_tefeat[feature].astype(str).values))    
        #
    #
    return pdf01A_trfeat, pdf02A_tefeat
###
pdf01A_trfeat, pdf02A_tefeat = encoder(pdf01A_trfeat, pdf02A_tefeat)
pdf01A_trfeat.groupby(feat_cols, as_index = False).agg(["count"])


is_monitor_training = True
is_show_helper = False
is_show_figure = False
def show_helper(function):
    if is_show_helper:
        pprint.pprint(help(function))
#
EPOCHS = 80
BATCH_SIZE = 128
###
include_tf_ph1 = False
include_tf_ph2 = False
include_tf_ph3 = True

data_phare = 5

### learning rate schedule
# following by: https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/
# should we do learning rate decay for adam optimizer 
# why should we use learning rate decay: https://www.coursera.org/lecture/deep-neural-network/learning-rate-decay-hjgIA
# read more: https://www.jeremyjordan.me/nn-learning-rate/
#
# triangular schedule with exponential decay 
from tensorflow.keras.callbacks import LearningRateScheduler
#
decay_factor = 0.75
step_size = 3
initial_lr = 0.05
def step_decay_schedule(initial_lr = initial_lr, 
        decay_factor = decay_factor, 
        step_size = step_size):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    #
    def plot_lr():
        rng = [i for i in range(EPOCHS)]
        y = [schedule(x) for x in rng]
        plt.plot(rng, y)
        print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
    #
    plot_lr()
    #
    return LearningRateScheduler(schedule)

lr_sched = step_decay_schedule()
#
show_helper(LearningRateScheduler)
#
early_stop = EarlyStopping(monitor = "val_loss",
    min_delta = 0, # minimum change in the monitored quantity to qualify as an improvement.
    patience = 10, # number of epochs with no impovement for training
    mode = "min", # if "min" mode, training will stop when quntity montier has stopped DECREASING
    restore_best_weights=True
    )
#
show_helper(EarlyStopping)

def _remove_higher_corr(pdf03_data:pd.DataFrame,
        cname: [],
        thresold= 0.9)->[]:
    print("""|_remove_higher_corr
        \n\t\t| len(cname) = {}
        \n\t\t|thresold = {}""".format(len(cname),thresold))
    #
    pdf03_cfeat_train_corr = pdf03_data[cname].corr()
    if is_show_figure:
        fig, axes = plt.subplots(1,1, figsize = (50,30))
        sns.heatmap(data = pdf03_cfeat_train_corr, ax=axes, cmap="mako")
    #
    tmp_selected_feat = np.full((len(cname),), True, dtype=bool)
    for i_idx in range(len(cname)):
        for j_idx in range(i_idx+1,len(cname)):
            if pdf03_cfeat_train_corr.iloc[i_idx,j_idx] >=thresold:
                if tmp_selected_feat[j_idx]:
                    tmp_selected_feat[j_idx] = False
    #
    cname_Sfeat = pdf03_data[cname].columns[tmp_selected_feat]
    # remove 
    print("\tremove cname feature: ", len(list(set(cname) - set(cname_Sfeat))))
    print("\tnew cname feature: ", len(cname_Sfeat))
    #
    return list(cname_Sfeat)
#
def preparing_data_4(pdf01A_trfeat, pdf01B_trtar, pdf02A_tefeat, outlier:int=2):
    """
    removed target that is less frquency 
    """
    pdf03_data = pd.concat([pdf01A_trfeat, pdf02A_tefeat])
    new_features = _remove_higher_corr(pdf03_data, feat_cols, thresold = 0.95)
    ###
    X = pdf01A_trfeat[new_features].values
    y = pdf01B_trtar[target_cols].values
    X_test = pdf02A_tefeat[new_features].values
    #
    unique_rows, unique_counts = np.unique(y, return_counts=True, axis=0)
    #
    pdf10_dict_hasing = pd.DataFrame({
        'row': [i for i in unique_rows],
        'class_count': list(unique_counts)
    })
    #
    pdf10_dict_hasing.sort_values(by='class_count', ascending=False, inplace=True)
    pdf10_dict_hasing.reset_index(drop=True, inplace=True)
    #
    row_to_class = {}
    class_to_row = {}
    for i, df_row in pdf10_dict_hasing.iterrows():
        row_to_class[tuple(df_row.row)] = i
        class_to_row[i] = df_row.row
    #
    y_classes = np.array([row_to_class[tuple(i)] for i in y])
    #
    if is_show_figure:
        pdf10_dict_hasing.class_count.plot(figsize=(13,6), logy=True, 
            title='Histogram of label combinations')
        plt.xlabel('Label combination no')
        plt.ylabel('Log Count')
        plt.show()
        # Find which classes are the least popular.
    outlier_classes = pdf10_dict_hasing[pdf10_dict_hasing.class_count < outlier].index
    print("remove : ", len(outlier_classes))
    # Find the index of the train labels where there are no unpopular classes
    filtered_idx = [i for i,x in enumerate(y_classes) if x not in outlier_classes]
    # Filter the train set without having unpopular classes
    X = X[filtered_idx]
    y = y[filtered_idx]
    #
    return X, X_test,y



def tf_model_2(layer_size = 2048, input_shape= 875):
    #
    model = tf.keras.Sequential([
        
        layers.BatchNormalization(input_shape=(input_shape,)),
        layers.Dropout(0.5),
        #
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(
            layers.Dense(layer_size//2, activation='elu')),
        layers.Dropout(0.5),
        #
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(
            layers.Dense(layer_size//4, activation='elu')),
        layers.Dropout(0.5),
        #
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(
            layers.Dense(206, activation=None)),
        ])
    #
    optimizer = tfa.optimizers.Lookahead(optimizers.Adam())
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        return tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    #
    model.compile(
        optimizer = optimizer,
        loss =loss_fn,
        metrics = ["accuracy"]
        )
    #
    if is_show_figure:
        model.summary()
    #
    return model
#
def tf_model_3(layer_size = 2048, input_shape= 875):
    from keras.regularizers import l2
    #
    model = tf.keras.Sequential([
        layers.BatchNormalization(input_shape=(input_shape,)),
        layers.Dropout(0.5),
        layers.Dense(layer_size//4, activation = "relu"),
        #
        layers.BatchNormalization(),
        layers.Dense(layer_size//8, activation = "relu",kernel_regularizer = l2(0.0005)),
        #
        layers.BatchNormalization(),
        layers.Dense(206, activation = "sigmoid"),
        #
        ])
    #
    model.compile(
        optimizer = "adam",
        loss ="binary_crossentropy",
        metrics = ["accuracy"]
        )
    #
    if is_show_figure:
        model.summary()
    #
    return model
#
def tf_model_1(layer_size = 2048, input_shape= 875):
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(1,input_shape)),
        #
        layers.Dense(layer_size//2, activation = "relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        #
        layers.Dense(layer_size//4, activation = "relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        #
        layers.Dense(layer_size//4, activation = "relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        #
        layers.Dense(206, activation = "sigmoid"),
        ])
    #
    model.compile(
        optimizer = "adam",
        loss ="binary_crossentropy",
        metrics = ["accuracy"]
        )
    #
    if is_show_figure:
        model.summary()
    #
    return model
#
def log_loss_metric(vals,preds):
    from sklearn.metrics import log_loss
    score = log_loss(np.ravel(vals),np.ravel(preds))
    print('Validation log loss score: {}'.format(score))
    #
    return score
#
def predict(model, X_test):
    if "tensorflow" in str(type(model)): 
        preds = np.array(model.predict(X_test).astype("float64"))
        #
    return preds
    #
def fit_model(model, X_train, X_valid, 
              y_train, y_valid,val_pred_lloss, options):
    import time
    start = time.time()
    if 'tensorflow' in str(type(model)):
        callbacks = options.get("callbacks")
        #
        history = model.fit(
            X_train,
            y_train,
            epochs = EPOCHS,
            verbose = 0 ,
            batch_size = BATCH_SIZE,
            callbacks = callbacks,
            validation_data = (X_valid, y_valid)
        )
        i_lloss_score = log_loss_metric(y_valid,predict(model, X_valid))
        val_pred_lloss.append(i_lloss_score)
        #
        if is_monitor_training:
            if "EarlyStopping" in str(callbacks):
                fig, ax = plt.subplots()
                #
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.title('model accuracy - TENSORFLOW')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()
                # summary history for loss 
                fig, ax = plt.subplots()
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss - TENSORFLOW')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()
        
    else: 
        model.fit(X_train, y_train)
#     print("total time stpending for fit model: ", time.time() - start)
    #
    return None
#
def preparing_data(pdf01A_trfeat,pdf01B_trtar, pdf02A_tefeat):
    X = pdf01A_trfeat.drop(columns = ["sig_id"]).values
    X_test = pdf02A_tefeat.drop(columns = ["sig_id"]).values
    y = pdf01B_trtar.drop(columns = ["sig_id"]).values
    return X, X_test, y
#
def preparing_data_2(pdf01A_trfeat, pdf01B_trtar, pdf02A_tefeat, outlier:int=6):
    """
    removed target that is less frquency 
    """
    X = pdf01A_trfeat[feat_cols].values
    y = pdf01B_trtar[target_cols].values
    X_test = pdf02A_tefeat[feat_cols].values
    #
    unique_rows, unique_counts = np.unique(y, return_counts=True, axis=0)
    #
    pdf10_dict_hasing = pd.DataFrame({
        'row': [i for i in unique_rows],
        'class_count': list(unique_counts)
    })
    #
    pdf10_dict_hasing.sort_values(by='class_count', ascending=False, inplace=True)
    pdf10_dict_hasing.reset_index(drop=True, inplace=True)
    #
    row_to_class = {}
    class_to_row = {}
    for i, df_row in pdf10_dict_hasing.iterrows():
        row_to_class[tuple(df_row.row)] = i
        class_to_row[i] = df_row.row
    #
    y_classes = np.array([row_to_class[tuple(i)] for i in y])
    #
    if is_show_figure:
        pdf10_dict_hasing.class_count.plot(figsize=(13,6), logy=True, 
            title='Histogram of label combinations')
        plt.xlabel('Label combination no')
        plt.ylabel('Log Count')
        plt.show()
        # Find which classes are the least popular.
    outlier_classes = pdf10_dict_hasing[pdf10_dict_hasing.class_count < outlier].index
    # Find the index of the train labels where there are no unpopular classes
    filtered_idx = [i for i,x in enumerate(y_classes) if x not in outlier_classes]
    # Filter the train set without having unpopular classes
    X = X[filtered_idx]
    y = y[filtered_idx]
    #
    return X, X_test,y
#
def preparing_data_3( pdf01A_trfeat,pdf01B_trtar, pdf02A_tefeat , cname_type = "g-"):
    ignored_cols = ["cy_type","cp_type","cp_dose"]
    cname_feats = [i_cname for i_cname in pdf01A_trfeat.columns if i_cname.startswith(cname_type)]
    X = pdf01A_trfeat[ignored_cols + cname_feats].values
    X_test = pdf02A_tefeat[ignored_cols + cname_feats].values
    y = pdf01B_trtar.drop(columns = ["sig_id"]).values
    return X, X_test, y,
#
#
def run_model(model, X_train, X_valid, 
        y_train, y_valid,X_test,val_pred_lloss, options_tf):
    # fit and evaluate on test and valid set
    fit_model(model, X_train, X_valid, y_train, y_valid,val_pred_lloss, options_tf)
    #
    y_pred = predict(model, X_test)
    #
    return model, y_pred
#
options_tf = {
    "callbacks": [lr_sched, early_stop],
}
num_split = 5
def kfold_test_model(pdf01A_trfeat, pdf01B_trtar, 
        pdf02A_tefeat,num_split = num_split,
        options_tf = options_tf):
    #
    seed = 12
    kfold = KFold(n_splits=num_split, random_state = seed, shuffle = True)
    if data_phare == 1:
        X, X_test,y = preparing_data(pdf01A_trfeat, pdf01B_trtar, pdf02A_tefeat)
    elif data_phare == 2:
        X, X_test,y = preparing_data_2(pdf01A_trfeat, pdf01B_trtar, pdf02A_tefeat)
    # if data_phare == 3:
    #     X, X_test,y = preparing_data_3(pdf01A_trfeat, pdf01B_trtar, pdf02A_tefeat, "g-")
    # if data_phare == 4:
    #     X, X_test,y = preparing_data_3(pdf01A_trfeat, pdf01B_trtar, pdf02A_tefeat, "c-")
    #
    elif data_phare ==5 :
        X, X_test,y = preparing_data_4(pdf01A_trfeat, pdf01B_trtar, pdf02A_tefeat)

    val_pred_lloss = []
    final_pred = []
    #
    final_pred_ph1 = []
    val_pred_lloss_ph1 = []
    #
    final_pred_ph2 = []
    val_pred_lloss_ph2 = []

    for fold, (train_index, valid_index) in enumerate(kfold.split(X, y)):
        print("Include tf: ", fold)
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        if include_tf_ph1:
            print("\tinclude_tf_ph1: ")
            model_1 = tf_model_1(input_shape = X_train.shape[1])
            model_1,y_pred = run_model(model_1, X_train, X_valid, 
                y_train, y_valid, X_test, val_pred_lloss_ph1 ,options_tf)
            final_pred_ph1.append(y_pred)
        if include_tf_ph2:
            print("\tinclude_tf_ph2: ")
            model_1 = tf_model_2(input_shape = X_train.shape[1])
            model_1,y_pred = run_model(model_1, X_train, X_valid, 
                y_train, y_valid, X_test, val_pred_lloss_ph2 ,options_tf)
            final_pred_ph2.append(y_pred)
        #
        if include_tf_ph3:
            print("\tinclude_tf_ph3: ")
            model_1 = tf_model_3(input_shape = X_train.shape[1])
            model_1,y_pred = run_model(model_1, X_train, X_valid, 
                y_train, y_valid, X_test, val_pred_lloss_ph2 ,options_tf)
            final_pred_ph2.append(y_pred)
    final_pred = final_pred_ph1 + final_pred_ph2
    val_pred_lloss = val_pred_lloss_ph1 + val_pred_lloss_ph2
    print("MEAN LLOSS VAL-SET: ", np.mean(val_pred_lloss))
    if include_tf_ph1:
        print("\tinclude_tf_ph1: ", np.mean(val_pred_lloss_ph1))
    if include_tf_ph2:
        print("\tinclude_tf_ph2: ", np.mean(val_pred_lloss_ph2))

    return final_pred
### Ensemble final predictions
final_pred = kfold_test_model(pdf01A_trfeat, pdf01B_trtar, 
        pdf02A_tefeat,num_split = 10,
        options_tf = options_tf)
def submission(pdf02B_tetar, final_pred):
    print('Ensembling final predictions')
    final_predictions = np.mean(np.array(final_pred),axis=0)
    print('Done')
    pdf02B_tetar.iloc[:,1:] = final_predictions
    pdf02B_tetar.to_csv('submission.csv',index=False)
    return None
submission(pdf02B_tetar, final_pred)
#### forcuse on Feature engineering and label proccessing
# The idea after read https://www.kaggle.com/pathofdata/nn-with-skf-strategy






















