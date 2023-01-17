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
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



SEED = 2145



# Feature engineering influenced by this notebook:

# https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial



def concat_df(train_data, test_data):

    # Returns a concatenated df of training and test set

    return 



def divide_df(all_data):

    # Returns divided dfs of training and test set

    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)



df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')

dfx = pd.concat([df_train, df_test], sort=True).reset_index(drop=True)



age_by_pclass_sex = dfx.groupby(['Sex', 'Pclass']).median()['Age']



# Filling the missing values in Age with the medians of Sex and Pclass groups

dfx['Age'] = dfx.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))



# Filling the missing values in Embarked with S

dfx['Embarked'] = dfx['Embarked'].fillna('S')



median_fare = dfx.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]

dfx['Fare'] = dfx['Fare'].fillna(median_fare)



dfx['Title'] = dfx['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

dfx['Is_Married'] = 0

dfx['Is_Married'].loc[dfx['Title'] == 'Mrs'] = 1



dfx = dfx.drop(['Name','Ticket','PassengerId','Cabin','Title'], axis=1)
X = dfx.loc[:890]

X_test = dfx.loc[891:]



labels = X['Survived']

X = X.drop(['Survived'], axis=1)



# Restrict the train data set to Numeric variables only

dt = pd.DataFrame(X.dtypes)

dt = dt.reset_index()

dt.columns = ['Column', 'Dtype']



### Handle Numeric Data

num_vars = np.array(dt[dt['Dtype']!='object']['Column'])

num_vars = num_vars[num_vars!='Parch']

num_vars = num_vars[num_vars!='Pclass']

num_vars = num_vars[num_vars!='SibSp']

num_vars = num_vars[num_vars!='Is_Married']

trainx_num = X.loc[:,num_vars]

testx_num = X_test.loc[:,num_vars]



### Handle Categorical Data

cat_vars = np.array(dt[dt['Dtype']=='object']['Column'])

# add day and hour here

cat_vars = np.append(cat_vars,'Parch')

cat_vars = np.append(cat_vars,'Pclass')

cat_vars = np.append(cat_vars,'SibSp')

cat_vars = np.append(cat_vars,'Is_Married')



trainx_cat = X.loc[:,cat_vars]

testx_cat = X_test.loc[:,cat_vars]



for varz in cat_vars:

    series = pd.value_counts(trainx_cat[varz])

    unique_elements = pd.concat([trainx_cat[varz],testx_cat[varz]]).unique().tolist()

    trainx_cat[varz] = trainx_cat[varz].astype('category').cat.set_categories(unique_elements)

    testx_cat[varz] = testx_cat[varz].astype('category').cat.set_categories(unique_elements)



# get final category dataset

trainx_cat = pd.get_dummies(trainx_cat, drop_first = True)

testx_cat = pd.get_dummies(testx_cat, drop_first = True)



# combine num and cat dataframes

df = pd.concat([trainx_num, trainx_cat], axis=1)

X_test = pd.concat([testx_num, testx_cat], axis=1)
X_train0, X_test, y_train0, y_test = train_test_split(df, labels, test_size=0.25, stratify=labels, random_state=33897)



X_train0, X_val, y_train0, y_val = train_test_split(X_train0, y_train0, test_size=0.10, stratify=y_train0, random_state=33897)



sc = StandardScaler()

sc.fit(X_train0)



X_train0 = sc.transform(X_train0)

X_val = sc.transform(X_val)

X_test = sc.transform(X_test)
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



        self.model.compile(optimizer=self.opt, loss=loss_func,metrics=['accuracy'])

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
layer_array = []

layer_array2 = []



for k in [48,64]:

    layer_array.append(tuple([k]*3))



lenq = len(layer_array)



for j in range(lenq):

    for i in range(8):

        tempo = list(layer_array[j])

        tempo[1] = tempo[1]*4*(i+1)

        tempo[2] = tempo[2]*(i+1)

        layer_array2.append(tempo)



layer_array2[0] = tuple(layer_array2[0])

layer_array2



print("To be Iterated upon: ",layer_array2)

print("----------------------------------------------")

print(" ")
X_train = X_train0

y_train = y_train0



Xinput_dimension = X_train.shape[1]



# Define Keras parameters

Xwt = 2

Xpt = 10

Xep = 50

Xbnorm = 1

Xverbose = 0

Xpms = 0

Xactivation = 'relu'

Xbsz = 2

X_use_weighted_loss = 0



for Xlayer in layer_array2:

    

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



    print("*********************")

    print("For Layer Config: ",Xlayer)

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