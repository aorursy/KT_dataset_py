import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

from prettytable import PrettyTable

from keras import optimizers

import collections

from keras_tqdm import TQDMNotebookCallback, TQDMCallback
label_paths = [

    "../input/data-files/y_train_smpl_0.csv",

    "../input/data-files/y_train_smpl_1.csv",

    "../input/data-files/y_train_smpl_2.csv",

    "../input/data-files/y_train_smpl_3.csv",

    "../input/data-files/y_train_smpl_4.csv",

    "../input/data-files/y_train_smpl_5.csv",

    "../input/data-files/y_train_smpl_6.csv",

    "../input/data-files/y_train_smpl_7.csv",

    "../input/data-files/y_train_smpl_8.csv",

    "../input/data-files/y_train_smpl_9.csv",

]
# Function we can use to load labels

def get_label (index=0, paths_array=label_paths):

    return pd.read_csv(paths_array [index])



# Load the features

data_train_all = pd.read_csv("../input/data-files/x_train_gr_smpl.csv")
from random import randint



# Use only a sample of the data to increase speed.

# Add a seed so that it returns the same numbers every time

random_int = randint(0, 1000)

def reduced_dataset(dataframe, size = 0.5):

    num_data = int(dataframe.shape[0]*size)

    np.random.seed(random_int)

    idx = np.arange(dataframe.shape[0])

    np.random.shuffle(idx)

    train_idx = idx[:num_data]

    

    return pd.DataFrame(dataframe.loc[train_idx].values)



data_train = reduced_dataset(data_train_all)
from sklearn.utils.random import sample_without_replacement



def sample_indices(test_size=0.25):

    n_samples = data_train.shape[0]

    train_size = round((1-test_size)*n_samples)

    test_size = n_samples - train_size

    

    all_indices = list(range(n_samples))

    train_indices = sample_without_replacement(n_population=data_train.shape[0], n_samples=train_size)

    test_indices = [x for x in all_indices if x not in train_indices]

    

    return train_indices, test_indices



test_size=0.25

train_idx, test_idx = sample_indices(test_size=test_size)
# Sets a threshold to the data and rounds it to 0 or 1. Useful for some metrics.

def set_threshold(data, threshold = 0.5):

    rounded = np.array([1 if x >= threshold else 0 for x in data])

    return rounded





def run_metrics(y_true, y_pred, threshold):

    y_pred_rounded = set_threshold(y_pred, threshold)



    roc_score = round(roc_auc_score(y_true, y_pred), 4)

    f_score = round(f1_score(y_true, y_pred_rounded), 4)

    recall = round(recall_score(y_true, y_pred_rounded), 4)

    precision = round(precision_score(y_true, y_pred_rounded), 4)

    accuracy = round(accuracy_score(y_true, y_pred_rounded), 4)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_rounded).ravel()

    

    

    return {

        "roc_score": roc_score,

        "f_score": f_score,

        "recall": recall,

        "precision": precision,

        "accuracy": accuracy,

        "TP": tp,

        "FP": fp}



def run_metrics_all(predictions_all, threshold = 0.5, labels_idx=[]):

    x = PrettyTable()

    x.field_names = ["", "Roc_AUC_score", "f score", "recall", "precision", "accuracy", "TP", "FP"]

    for i, (columnName, y_pred) in enumerate(predictions_all.iteritems()):

        y_true = reduced_dataset(get_label(index=i, paths_array=label_paths))

        

        if len(labels_idx) != 0:

            y_true = y_true.loc[labels_idx]

            

        metrics = run_metrics(y_true, y_pred, threshold=threshold)

        x.add_row([columnName, metrics["roc_score"], metrics["f_score"],

                   metrics["recall"], metrics["precision"], metrics["accuracy"], metrics["TP"], metrics["FP"]])

        

    print(x)
from sklearn.svm import LinearSVC



from sklearn.preprocessing import StandardScaler



clf = LinearSVC(random_state=1, max_iter=50)



regresion_predictions = pd.DataFrame()



for i in range(len(label_paths)):

    target = reduced_dataset(get_label(index=i, paths_array=label_paths))

    

    #Scale the data using standarization

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data_train.values)

    train_data, test_data = data_train.loc[train_idx], data_train.loc[test_idx]

    train_target, test_target = target.loc[train_idx], target.loc[test_idx]

    

    linear_clf = clf.fit(scaled_data, target.values.ravel())

    prediction = linear_clf.predict(test_data)

    regresion_predictions["y_train_smpl_{}".format(i)] = prediction 
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.metrics import binary_accuracy

from keras import backend as K

from sklearn import metrics

import tensorflow as tf



def create_model ():

    # define the keras model

    model = Sequential()

    model.add(Dense(120, input_dim=2304, activation='relu'))

    model.add(Dense(60, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))



    # Compile the model

    # For a binary classification problem

    sgd = optimizers.RMSprop(lr=0.001)



    model.compile(optimizer=sgd,

                  loss='binary_crossentropy',

                  metrics=['accuracy'])

    

    return model
# Optimiziers

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1,

                              patience=5, min_lr=1e-8, verbose=1)



early_stopper = EarlyStopping(monitor='val_loss', patience=10,

                              verbose=0, restore_best_weights=True)
from sklearn.model_selection import KFold



# Initialize 10 fold cross validation

n_splits = 10

random_state = np.random.seed(22135)



kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=False)
def train_NN_CV(label_paths=label_paths):

    results = pd.DataFrame()

    test_preds = pd.DataFrame()



    for i in range(len(label_paths)):

        print("\n\nRUNNING NET FOR y_train_smpl_{}".format(i))

        # oof -> Out of fold. One single vector with all the validation predictions to

        # then calculate the error upon this predictions.

        oof = np.zeros(data_train.shape[0])

        predictions_test = np.zeros(data_train.shape[0])

        

        target = reduced_dataset(get_label(index=i, paths_array=label_paths))



        # K-fold CV

        for epoch, (train_index, val_index) in enumerate(kf.split(data_train.values)):

            print("\nFOLD {}".format(epoch + 1))

            X_train, X_val = data_train.loc[train_index], data_train.loc[val_index]

            y_train, y_val = target.loc[train_index], target.loc[val_index]

            

            model = create_model()

                

            model.fit(X_train, y_train,

                   batch_size=60, epochs=35,

                   validation_data=(X_val, y_val),

                   verbose=1,

                   callbacks=[scheduler])

            

            

            # Predict on Val data

            prediction = model.predict(X_val.astype('float32')).squeeze()

            oof[val_index] = prediction

            predictions_test += (model.predict(data_train)/kf.n_splits).squeeze()

            

        test_preds["y_train_smpl_{}".format(i)] = predictions_test 

        results["y_train_smpl_{}".format(i)] = oof 

        print(run_metrics(target, predictions_test, threshold=0.5))

        

    return results, test_preds

predictions, predictions_test = train_NN_CV()
# benchmark for linear regression

# Metrics based on the predictions of test data (unseen data)

run_metrics_all(regresion_predictions, labels_idx=test_idx)
## Set a benchmark for all 1s

np_ones = np.ones(predictions.shape)

ones = pd.DataFrame(np_ones)



run_metrics_all(ones, threshold=0.5)
# Get results (predictions are the predictions on the valid set in each fold (unseen data))

run_metrics_all(predictions)
def create_model_1():

    # define the keras model

    model = Sequential()

    model.add(Dense(240, input_dim=2304, activation='relu'))

    model.add(Dense(120, activation='relu'))

    model.add(Dense(120, activation='relu'))

    model.add(Dense(60, activation='relu')) 

    model.add(Dense(60, activation='relu'))   

    model.add(Dense(30, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))



    # Compile the model

    # For a binary classification problem

    sgd = optimizers.RMSprop(lr=0.0001)



    model.compile(optimizer=sgd,

                  loss='binary_crossentropy',

                  metrics=['accuracy'])



    return model
def create_model_2():

    # define the keras model

    model = Sequential()

    model.add(Dense(240, input_dim=2304, activation='relu'))

    model.add(Dense(120, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(60, activation='relu')) 

    model.add(Dropout(0.5))

    model.add(Dense(30, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))



    # Compile the model

    # For a binary classification problem

    sgd = optimizers.RMSprop(lr=0.0001)



    model.compile(optimizer=sgd,

                  loss='binary_crossentropy',

                  metrics=['accuracy'])



    return model
def create_model_3():

    # define the keras model

    # define the keras model

    model = Sequential()

    model.add(Dense(240, input_dim=2304, activation='relu'))

    model.add(Dense(120, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(60, activation='relu')) 

    model.add(Dropout(0.5))

    model.add(Dense(30, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))



    # Compile the model

    # For a binary classification problem

    sgd = optimizers.SGD(lr=0.0001, momentum=0.9)



    model.compile(optimizer=sgd,

                  loss='binary_crossentropy',

                  metrics=['accuracy'])



    return model
# Same as 3 but with different momentum

def create_model_4():

    # define the keras model

    # define the keras model

    model = Sequential()

    model.add(Dense(240, input_dim=2304, activation='relu'))

    model.add(Dense(120, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(60, activation='relu')) 

    model.add(Dropout(0.5))

    model.add(Dense(30, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))



    # Compile the model

    # For a binary classification problem

    sgd = optimizers.SGD(lr=0.0001, momentum=0.1)



    model.compile(optimizer=sgd,

                  loss='binary_crossentropy',

                  metrics=['accuracy'])



    return model
# Optimiziers

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1,

                              patience=5, min_lr=1e-8, verbose=1)



early_stopper = EarlyStopping(monitor='val_loss', patience=10,

                              verbose=0, restore_best_weights=True)
from sklearn.model_selection import train_test_split

seed = np.random.seed(0)



def train_net(train_idx, test_idx, label_paths=label_paths, model_number=0):

    results = pd.DataFrame()



    for i in range(len(label_paths)):

        print("\nRUNNING NET FOR y_train_smpl_{}".format(i))

        #Scale the data using standarization

        scaler = StandardScaler()

        scaled_data = pd.DataFrame(scaler.fit_transform(data_train.values))

        

        target = reduced_dataset(get_label(index=i, paths_array=label_paths))

        train_X, test_X = scaled_data.loc[train_idx], scaled_data.loc[test_idx]

        train_y, test_y = target.loc[train_idx], target.loc[test_idx]

        

        if model_number==0:

            model = create_model()

        elif model_number==1:

            model = create_model_1()

        elif model_number == 2:

            model = create_model_2()

        elif model_number == 3:

            model = create_model_3()

            

        

        model.fit(train_X, train_y,

               batch_size=32, epochs=40,

               validation_data=(test_X, test_y),

               verbose=0,

               callbacks=[scheduler, early_stopper])

            

        # Predict on Val data

        prediction = model.predict(test_X.astype('float32')).squeeze()

        results["y_train_smpl_{}".format(i)] = prediction

    return results

        

predictions = train_net(train_idx, test_idx)

predictions.head()
run_metrics_all(predictions, labels_idx=test_idx)
# Run for 4000 test samples

test_size=0.35

train_idx, test_idx = sample_indices(test_size=test_size)

predictions_4000 = train_net(train_idx, test_idx)
run_metrics_all(predictions_4000, labels_idx=test_idx)
# Run for 9000 test samples

test_size=0.77

train_idx, test_idx = sample_indices(test_size=test_size)

predictions_9000 = train_net(train_idx, test_idx)
run_metrics_all(predictions_9000, labels_idx=test_idx)
test_size=0.35

train_idx, test_idx = sample_indices(test_size=test_size)

predictions1_4000 = train_net(train_idx, test_idx, model_number=1)
print("4000 test size for overfitting model")

run_metrics_all(predictions1_4000, labels_idx=test_idx)
test_size=0.77

train_idx, test_idx = sample_indices(test_size=test_size)

predictions1_9000 = train_net(train_idx, test_idx,  model_number=1)
print("90000 test size for overfitting model")

run_metrics_all(predictions1_9000, labels_idx=test_idx)
test_size=0.35

train_idx, test_idx = sample_indices(test_size=test_size)

predictions2_4000 = train_net(train_idx, test_idx, model_number=2)
print("4000 test size for overfitting model")

run_metrics_all(predictions2_4000, labels_idx=test_idx)
test_size=0.77

train_idx, test_idx = sample_indices(test_size=test_size)

predictions2_9000 = train_net(train_idx, test_idx,  model_number=2)
print("90000 test size for overfitting model")

run_metrics_all(predictions2_9000, labels_idx=test_idx)
test_size=0.35

train_idx, test_idx = sample_indices(test_size=test_size)

predictions3_4000 = train_net(train_idx, test_idx, model_number=3)
print("4000 test size for overfitting model")

run_metrics_all(predictions3_4000, labels_idx=test_idx)
test_size=0.77

train_idx, test_idx = sample_indices(test_size=test_size)

predictions3_9000 = train_net(train_idx, test_idx,  model_number=2)
print("90000 test size for overfitting model")

run_metrics_all(predictions3_9000, labels_idx=test_idx)
test_size=0.35

train_idx, test_idx = sample_indices(test_size=test_size)

predictions4_4000 = train_net(train_idx, test_idx, model_number=3)
print("4000 test size model with momentum")

run_metrics_all(predictions4_4000, labels_idx=test_idx)
test_size=0.77

train_idx, test_idx = sample_indices(test_size=test_size)

predictions4_9000 = train_net(train_idx, test_idx,  model_number=2)
print("90000 test size for overfitting model")

run_metrics_all(predictions4_9000, labels_idx=test_idx)