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
X_train = pd.read_csv('../input/kml2020/X_train.csv', encoding = 'cp949')

y_train = pd.read_csv('../input/kml2020/y_train.csv',encoding = 'cp949')

X_test = pd.read_csv('../input/kml2020/X_test.csv', encoding = 'cp949')

IDtest = X_test.cust_id.unique()



tr = pd.merge(X_train, X_test, how = 'outer')

tr.head()
data = pd.read_csv('../input/additional/train_test_ohe.csv', encoding = 'cp949')

OH_w2v = pd.read_csv('../input/additional/OH_w2v.csv', encoding = 'cp949')

seahong_w2v = pd.read_csv('../input/additional/seahong_w2v.csv', encoding = 'cp949')



data = pd.merge(data, OH_w2v, on = 'cust_id')

data = pd.merge(data, seahong_w2v, on = 'cust_id')



train = data.query('cust_id < 3500')

test = data.query('cust_id >= 3500')
X_train = train.copy()

X_test = test.copy()

train_test = data.copy()
X_train = train_test.query('cust_id < 3500')

X_test = train_test.query('cust_id >= 3500')

#y_train = pd.read_csv('y_train.csv')



#del X_train['cust_id']

#del X_test['cust_id']



X_train.drop('cust_id', axis = 1, inplace = True)

X_test.drop('cust_id', axis = 1, inplace = True)



tr_col = X_train.columns

te_col = X_test.columns



scaler = MinMaxScaler()

X_train = pd.DataFrame(scaler.fit_transform(np.array(X_train)))

X_train.columns = tr_col



X_test = pd.DataFrame(scaler.transform(np.array(X_test)))

X_test.columns = te_col
model = LogisticRegression(random_state=0)



# 각 특성과 타깃(class) 사이에 유의한 통계적 관계가 있는지 계산하여 특성을 선택하는 방법 

cv_scores = []





########### cv 바꿔줌 ###############

s_fold = StratifiedKFold(n_splits = 5, random_state = 0)



for p in tqdm(range(5,100,1)):

    X_new = SelectPercentile(percentile=p).fit_transform(X_train, y_train.gender)    

    cv_score = cross_val_score(model, X_new, y_train.gender, scoring='roc_auc', cv=s_fold).mean()

    cv_scores.append((p,cv_score))



# Print the best percentile

best_score = cv_scores[np.argmax([score for _, score in cv_scores])]

print(best_score)



# Plot the performance change with p

plt.plot([k for k, _ in cv_scores], [score for _, score in cv_scores])

plt.xlabel('Percent of features')

plt.grid()
# 과적합을 피하기 위해 최적의 p값 주변의 값을 선택하는게 더 나은 결과를 얻을 수 있다. 

selectp = SelectPercentile(percentile=best_score[0]).fit(X_train, y_train.gender)

X_train_sel = selectp.transform(X_train)

X_test_sel = selectp.transform(X_test)
# 학습데이터 70%, 평가데이터 30%로 데이터 분할

X_train, X_valid, y_train, y_valid = train_test_split(X_train_sel, y_train.gender, test_size=0.3, random_state=0)
# Set hyper-parameters for power mean ensemble 

N = 10

#p = 3.5

preds3 = []

aucs = []



for i in tqdm(range(N)):    

    #X_train, X_test = train, test

    

    ##### STEP 1: Randomize Seed

    SEED = np.random.randint(1, 10000)              

    random.seed(SEED)       

    np.random.seed(SEED)     

    if tf.__version__[0] < '2':  

        tf.set_random_seed(SEED)

    else:

        tf.random.set_seed(SEED)



    ##### STEP 2: Build a DNN Model



    # Define the Model architecture

    

    input = Input(shape=(X_train.shape[1],))

    

    x1 = Dense(1024, activation='relu')(input)

    x = Dropout(0.5)(x1)  # 노드 몇개 탈락시킬지

    x = Dense(512, activation = 'relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)

    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu')(x)

    x = Dense(64, activation='relu')(x)

    x = Dense(16, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x) # 출력층

    model = Model(input, output)



    # Train the Model

    model.compile(loss='binary_crossentropy', optimizer= RMSprop(lr=1e-4), metrics=[tf.keras.metrics.AUC()])

    

    

    #train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.2)

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),]

    history = model.fit(X_train, y_train, validation_split = 0.2, batch_size=512, epochs=150, callbacks=callbacks, 

                        shuffle=True, verbose=2)



    print(f'DNN learning curve {i+1}/{N}')

    plt.plot(history.history["loss"], label="train loss")

    plt.plot(history.history["val_loss"], label="validation loss")

    plt.legend()

    plt.title("Loss")

    plt.show()

    

    # Make Prediction

    auc = roc_auc_score(y_valid, model.predict(X_valid))

    aucs.append(auc)

    print('AUC', auc)

    

    preds3.append(model.predict(X_test_sel))
### Validate the Models

print('\nValidation Summary:')

aucs = pd.Series(aucs)

print(aucs.sort_values(ascending=False))

print('mean={:.5f}, std={:.3f}'.format(aucs.mean(), aucs.std()))    
# Gmean ensemble

THRESHOLD = 0.765  # Use only models whose AUC exceeds this value

#상위3개 기하평균

pred = 1

n = 0



for i in range(N):

    if aucs.iloc[i] > THRESHOLD:

        pred *= preds3[i]

        n += 1   

pred = pred**(1/n)



# Make a submission file

subid = pd.DataFrame(IDtest)

subid.columns = ['cust_id']

sub1 =pd.DataFrame(pred)

sub1.columns = ['gender']



submissions = pd.concat([subid, sub1], axis = 1)

submissions.to_csv('0.794_seed_ver3.csv', index=False, encoding ='cp949')