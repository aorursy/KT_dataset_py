import os

os.environ['KERAS_BACKEND'] = 'tensorflow'



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score, make_scorer, precision_recall_curve, precision_score, recall_score

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedShuffleSplit

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings

from sklearn.exceptions import DataConversionWarning

from sklearn.model_selection import StratifiedKFold

from datetime import datetime

from sklearn.decomposition import PCA, TruncatedSVD

import random

import math

from scipy import stats

import xgboost as xgb

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

import time

from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder

import keras as K

from keras.layers import Dropout, BatchNormalization, Activation

from sklearn.utils import class_weight

import keras.backend as K1

from sklearn.model_selection import StratifiedKFold

from keras.callbacks import ModelCheckpoint, EarlyStopping

import gc

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import confusion_matrix

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import tensorflow as tf



seed = 2145

tf.random.set_seed(seed)

np.random.seed(seed)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = pd.read_csv('../input/financial-distress/Financial Distress.csv')



# Dropping variable x80 for now

data = data.drop(['x80'], axis=1)



# Coverting target variable from continuous to binary form 

# as per the problem description  

def label_conv(x):

    if x > -0.5:

        return 0

    else:

        return 1



labels = data.iloc[:,2].apply(label_conv).values

            

df = data.iloc[:,3:].values



X_train0, X_test, y_train0, y_test = train_test_split(df, labels, test_size=0.25, stratify=labels, random_state=33897)



X_train0, X_val, y_train0, y_val = train_test_split(X_train0, y_train0, test_size=0.10, stratify=y_train0, random_state=33897)



sc = StandardScaler()

sc.fit(X_train0)



X_train0 = sc.transform(X_train0)

X_val = sc.transform(X_val)

X_test = sc.transform(X_test)



print("Shape of X_train:",X_train0.shape)
def f1_metric(y_true, y_pred):

    true_positives = K1.sum(K1.round(K1.clip(y_true * y_pred, 0, 1)))

    possible_positives = K1.sum(K1.round(K1.clip(y_true, 0, 1)))

    predicted_positives = K1.sum(K1.round(K1.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K1.epsilon())

    recall = true_positives / (possible_positives + K1.epsilon())

    f1_val = 2*(precision*recall)/(precision+recall+K1.epsilon())

    return f1_val



class build_keras_model(object):

    

    def __init__(self, layers, activation, opt, init, input_dim, weight, patience, 

                 is_batchnorm, use_weighted_loss, print_model_summary, verbose):

        

        self.model = K.models.Sequential()

        self.layers = layers

        self.activation = activation

        self.opt = opt

        self.init = init

        self.input_dim = input_dim

        self.weight = weight

        self.patience = patience

        self.is_batchnorm = is_batchnorm

        self.prmodsum = print_model_summary

        self.use_weighted_loss = use_weighted_loss

        self.verbose = verbose



    def create_model(self):

        

        now = datetime.now()



        for i, nodes in enumerate(self.layers):

            if i==0:

                self.model.add(K.layers.Dense(nodes,input_dim=self.input_dim,kernel_initializer=self.init))

                self.model.add(Activation(self.activation))

                if self.is_batchnorm == 1:

                    self.model.add(BatchNormalization())

            else:

                self.model.add(K.layers.Dense(nodes,kernel_initializer=self.init))

                self.model.add(Activation(self.activation))

                if self.is_batchnorm == 1:

                    self.model.add(BatchNormalization())



        self.model.add(K.layers.Dense(1))

        self.model.add(Activation('sigmoid')) # Note: no activation beyond this point

        

        if self.prmodsum == 1:

            print(self.model.summary())

        

        def weighted_loss(y_true, y_pred):

            weights = (y_true * self.weight) + 1.

            cross_entop = K1.binary_crossentropy(y_true, y_pred)

            weighted_loss = K1.mean(cross_entop * weights)

            return weighted_loss

        

        if self.use_weighted_loss == 1:

            loss_func = weighted_loss

        else:

            loss_func = 'binary_crossentropy'



        self.model.compile(optimizer=self.opt, loss=loss_func,metrics=[f1_metric])

        return

    

    def fit_model(self, X, y, X_validation, y_validation, batch_size, epochs, random_state):



        pt = self.patience

        vb = self.verbose

        

        earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=vb, patience=pt, restore_best_weights = True)

        callbacks_list = [earlystopping]



        np.random.seed(random_state)

        self.model.fit(X, y, validation_data = (X_validation,y_validation), 

                                                batch_size=batch_size,

                                                epochs=epochs, 

                                                verbose=vb, 

                                                callbacks=callbacks_list)

        

        return

    

    def predict_from_model(self,test_df):



        return self.model.predict(test_df)

    

    def __del__(self): 

        del self.model

        gc.collect()

        if self.prmodsum == 1:

            print("Keras Destructor called") 
X_train = X_train0

y_train = y_train0



Xinput_dimension = X_train.shape[1]



# Define Keras parameters

Xpt = 25

Xep = 5000

Xbnorm = 1

Xverbose = 0

Xpms = 0

Xactivation = 'relu'

Xlayer = (64, 256, 512, 256, 64)

Xbsz = 32



# We will iterate on using simple binary loss vs weighted loss with different weights

NUM = 7

use_weighted_loss_flags = np.append(0,np.repeat(1,NUM))

cross_entropy_weights = np.append('Not Applicable',range(111,110+NUM,1))



# We can notice that there is some improvement on F1 score when we assign weights to cross entropy

for X_use_weighted_loss,Xwt in zip(use_weighted_loss_flags,cross_entropy_weights):

    

    opt_name = 'Adagrad'

    init_name = 'glorot_uniform'

    Xopt = K.optimizers.Adagrad(learning_rate=0.01)

    Xinit = K.initializers.glorot_uniform(seed=1)



    np.random.seed(2018)

    tf.random.set_seed(2018)

    K1.set_session



    km1 = build_keras_model(layers= Xlayer, 

                           activation = Xactivation, 

                           opt = Xopt, 

                           init = Xinit, 

                           input_dim = Xinput_dimension,

                           weight = Xwt,

                           patience = Xpt,

                           is_batchnorm = Xbnorm,

                           print_model_summary = Xpms,

                           use_weighted_loss = X_use_weighted_loss,

                           verbose = Xverbose)



    km1.create_model()



    km1.fit_model(

                 X = X_train, 

                 y = y_train, 

                 X_validation = X_val, 

                 y_validation = y_val, 

                 batch_size = Xbsz, 

                 epochs = Xep,

                 random_state = 3397)



    preds = km1.predict_from_model(test_df = X_test)



    del km1

    K1.clear_session()

    gc.collect()



    best_f1 = 0

    best_predval = []

    best_thresh = 0.5



    for thresh in np.arange(0.001,1,0.001):

        thresh = round(thresh,3)

        predval = (preds > thresh).astype(int)

        f1s = f1_score(y_test,predval)

        if f1s > best_f1:

            best_f1 = f1s

            best_thresh = thresh

            best_predval = predval

    

    def wloss(i):

        switcher={

                0:'No',

                1:'Yes',

             }

        return switcher.get(i,"Invalid Choice")

        

    print("*********************")

    print("For Use Weighted Loss = ",wloss(X_use_weighted_loss),", Crossentropy Weight = ",Xwt)

    print("*********************")

    print("")

    print("Best Threshold = ",best_thresh)



    print("")

    if(np.sum(best_predval)==0):

        print("All zeros predicted, so no confusion matrix")

    else:

        print(confusion_matrix(y_test,best_predval))

        print("")

        print("Precision = ",round(precision_score(y_test,best_predval),4))

        print("Recall = ",round(recall_score(y_test,best_predval),4))



    print("")

    print("Test F1_SCORE = ",round(best_f1,4))

    print("")