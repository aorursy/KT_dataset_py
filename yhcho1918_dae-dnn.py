# Data Wrangling

import pandas as pd

from pandas import Series, DataFrame

import numpy as np



# Visualization

import seaborn as sns

import matplotlib.pylab as plt

from matplotlib import font_manager, rc



# Preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.decomposition import PCA

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.compose import make_column_transformer

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import FeatureUnion



from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import SelectPercentile

from sklearn.preprocessing import PolynomialFeatures



# Modeling

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



from sklearn.ensemble import VotingClassifier

from vecstack import stacking

from scipy.stats.mstats import gmean



# Evaluation

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import average_precision_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import LeaveOneOut

from sklearn.model_selection import ShuffleSplit



# Utility

import os

import time

import random

import warnings; warnings.filterwarnings("ignore")

from IPython.display import Image

import pickle

from itertools import combinations

import gc

from tqdm import tqdm

import platform

import datetime



# For DNN modeling

import tensorflow as tf

import keras

from keras import backend as K

from keras.layers import * #Input, Dense

from keras.models import * #Model

from keras.optimizers import *

from keras.initializers import *

from keras.regularizers import *

from keras.utils.np_utils import *

from keras.utils.vis_utils import * #model_to_dot

from keras.callbacks import EarlyStopping
tr_train = pd.read_csv('../input/kml2020/X_train.csv', encoding='cp949')

tr_test = pd.read_csv('../input/kml2020/X_test.csv', encoding='cp949')

y_train = pd.read_csv('../input/kml2020/y_train.csv').gender

IDtest = tr_test.cust_id.unique()



tr_train.head()
def make_goods_dict(df_train, df_test):

    

    data = pd.concat([df_train,df_test], axis = 0)

    goods_list = np.array(df_train.query('gds_grp_nm != "상품군미지정"')['goods_id'].unique())

    temp = df_train.query('gds_grp_nm != "상품군미지정" and gds_grp_nm != "기타" and goods_id in @goods_list')\

    [['goods_id','gds_grp_nm','gds_grp_mclas_nm' ]].drop_duplicates()

    

    goods_dict = {}

    for index in range(temp.shape[0]):

        goods_dict[temp.iloc[index][0]] = [temp.iloc[index][1], temp.iloc[index][2]]

        

    return goods_dict
def goods_tag(df, goods_dict):

    gds_grp_nm = []

    gds_grp_mclas_nm = []

    

    for index in range(df.shape[0]):

        if df.iloc[index]['gds_grp_nm'] == '상품군미지정' :

            try : 

                key = df.iloc[index]['goods_id']

                nm_mclas = goods_dict.get(key)

                gds_grp_nm.append(nm_mclas[0])

                gds_grp_mclas_nm.append(nm_mclas[1])

            except:

                gds_grp_nm.append(df.iloc[index]['gds_grp_nm'])

                gds_grp_mclas_nm.append(df.iloc[index]['gds_grp_mclas_nm'])

        else :

            gds_grp_nm.append(df.iloc[index]['gds_grp_nm'])

            gds_grp_mclas_nm.append(df.iloc[index]['gds_grp_mclas_nm'])

            

    df = pd.DataFrame({'gds_grp_nm' :gds_grp_nm , 'gds_grp_mclas_nm' : gds_grp_mclas_nm})

    return df
def replace_col(df, df_goods_map):

    # 열 대체

    df = df.drop(['gds_grp_nm','gds_grp_mclas_nm'],axis = 1)

    df = pd.concat([df, df_goods_map ], axis = 1)

    df = df[['cust_id', 'tran_date', 'store_nm', 'goods_id','gds_grp_nm','gds_grp_mclas_nm', 'amount']]

    return df
def goods_operator(df_train,df_test):

    # 위의 함수들을 자동적으로 실행한다.

    goods_dict = make_goods_dict(df_train, df_test)

    df_train = replace_col(df_train, goods_tag(df_train, goods_dict))

    df_test = replace_col(df_test, goods_tag(df_test, goods_dict))

    return df_train, df_test
%%time



tr_train_new, tr_test_new = goods_operator(tr_train, tr_test)
features = ['goods_id', 'gds_grp_nm', 'gds_grp_mclas_nm']



tr_all = pd.concat([tr_train_new, tr_test_new])

train = []

test = []



for f in features:

    for d,q in zip([train, test], ['cust_id not in @IDtest', 'cust_id in @IDtest']):

        d.append(pd.pivot_table(tr_all, index='cust_id', columns=f, values='amount',

                                aggfunc=lambda x: np.where(len(x) >=1, 1, 0), fill_value=0)                 

                 .reset_index()

                 .query(q)

                 .drop(columns=['cust_id']).values)

 

train, test = np.hstack(train),  np.hstack(test)
# Set hyper-parameters for power mean ensemble 

N = 20

p = 3.5

preds = []

aucs = []



for i in tqdm(range(N)):    

    X_train, X_test = train, test



    ##### STEP 1: Randomize Seed

    SEED = int(time.time() * 10000000 % 10000000) #np.random.randint(1, 10000)              

    random.seed(SEED)       

    np.random.seed(SEED)     

    if tf.__version__[0] < '2':  

        tf.set_random_seed(SEED)

    else:

        tf.random.set_seed(SEED)



    ##### STEP 2: Build DAE #####

    

    # Define the encoder dimension

    encoding_dim = 128



    # Input Layer

    input_dim = Input(shape = (X_train.shape[1], ))



    # Encoder Layers

    noise = Dropout(0.5)(input_dim) # for Denoising

    encoded1 = Dense(512, activation = 'relu')(noise)

    encoded2 = Dense(256, activation = 'relu')(encoded1)

    encoded3 = Dense(128, activation = 'relu')(encoded2)

    encoded4 = Dense(encoding_dim, activation = 'relu')(encoded3)



    # Decoder Layers

    decoded1 = Dense(128, activation = 'relu')(encoded4)

    decoded2 = Dense(256, activation = 'relu')(decoded1)

    decoded3 = Dense(512, activation = 'relu')(decoded2)

    decoded4 = Dense(X_train.shape[1], activation = 'linear')(decoded3)



    # Combine Encoder and Deocder layers

    autoencoder = Model(inputs = input_dim, outputs = decoded4)



    # Compile the model

    autoencoder.compile(optimizer = 'adam', loss = 'mse')



    # Train the model

    history = autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, 

                              shuffle=True, validation_data=(X_test,X_test), verbose=0)



    print(f'DAE learning curve {i+1}/{N}')

    plt.plot(history.history["loss"], label="train loss")

    plt.plot(history.history["val_loss"], label="validation loss")

    plt.legend()

    plt.title("Loss")

    plt.show()



    ##### STEP 3: Reduce Dimension #####

        

    # Use a middle Bottleneck Layer to Reduce Dimension

    model = Model(inputs=input_dim, outputs=encoded4)

    X_train = model.predict(X_train)

    X_test = model.predict(X_test)



    ##### STEP 4: Build a DNN Model



    # Define the Model architecture

    max_features = X_train.shape[1]



    # Define the Model architecture

    model = Sequential()

    model.add(Dense(32, activation='relu', input_shape=(max_features,), kernel_regularizer=l2(0.01)))   

    model.add(Dropout(0.7))

    model.add(Dense(16, activation='relu'))

    model.add(Dropout(0.7))

    model.add(Dense(1, activation='sigmoid'))



    # Train the Model

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(curve='ROC', name='roc_auc')])

    train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.2)

    history = model.fit(train_x, train_y, epochs=100, batch_size=256, 

                        validation_data=(valid_x,valid_y), callbacks=[tf.keras.callbacks.EarlyStopping(monitor= 'val_roc_auc', patience=15, mode='max')], verbose=0)



    print(f'DNN learning curve {i+1}/{N}')

    plt.plot(history.history["loss"], label="train loss")

    plt.plot(history.history["val_loss"], label="validation loss")

    plt.legend()

    plt.title("Loss")

    plt.show()

    

    # Make Prediction

    auc = roc_auc_score(valid_y, model.predict(valid_x).flatten())

    aucs.append(auc)

    print('AUC', auc)

    preds.append(model.predict(X_test).flatten())     
### Validate the Models

print('\nValidation Summary:')

aucs = pd.Series(aucs)

print(aucs.sort_values(ascending=False))

print('mean={:.5f}, std={:.3f}'.format(aucs.mean(), aucs.std()))   
# Power mean ensemble

THRESHOLD = 0.78  # Use only models whose AUC exceeds this value



sns.heatmap(pd.DataFrame(preds).filter(aucs[aucs.sort_values(ascending=False) > THRESHOLD].index.to_list(), axis=0).T.corr(), annot=True)

plt.show()



pred = 0

n = 0

for i in range(N):

    if aucs.iloc[i] > THRESHOLD:

        pred = pred + preds[i]**p 

        n += 1

pred = pred / n    

pred = pred**(1/p)



# Make a submission file



submissions = pd.concat([pd.Series(IDtest, name="cust_id"), pd.Series(pred, name="gender")] ,axis=1)

submissions.to_csv('submission_DAE(batchsize_256).csv', index=False)