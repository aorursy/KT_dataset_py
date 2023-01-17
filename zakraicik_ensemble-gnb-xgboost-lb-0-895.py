import pandas as pd

import numpy as np

import os

import sys

from xgboost.sklearn import XGBClassifier

from sklearn.metrics import roc_auc_score , make_scorer,accuracy_score

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

import seaborn as sns

from tqdm import tqdm

from sklearn.externals import joblib

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from keras.utils.np_utils import to_categorical

from sklearn.naive_bayes import GaussianNB

from keras.models import Sequential

from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D

from keras.optimizers import Adam ,RMSprop

from keras.layers.core import  Lambda , Dense, Flatten, Dropout  

from keras.callbacks import Callback

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

import operator
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
x = train.iloc[:,2:]

y = train['target']
x_train, x_test, y_train,y_test = train_test_split(x,

                                                    y,

                                                    random_state = 21,

                                                    test_size = 0.15

                                                  )
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
for i in range(x_train.shape[1]):

    ax1 = sns.distplot(x_train[(y_train==0),i])

    ax2 = sns.distplot(x_train[(y_train==1),i])

    plt.show()
##for i in range(x_train.shape[1]):

  ##  y_train = y_train[abs(x_train[:,i]) <= 2.5]

   ## x_train = x_train[abs(x_train[:,i]) <= 2.5]
neg = len(y_train)-sum(y_train)

pos = sum(y_train)

scale_pos_weight  = float(neg/pos)
XGB = XGBClassifier(scale_pos_weight=scale_pos_weight,

                        objective='binary:logistic',

                        random_state= 21,

                        tree_method = 'gpu_hist',

                        learning_rate = 0.1, ## From initial gridsearch

                        n_estimators = 1000 ,  ## From initial gridsearch

                        tree_depth= 3     ## From initial gridsearch

                    )
XGB.fit(x_train, y_train)
y_preds = XGB.predict_proba(x_test)
probs_pos_XGB  = []

for pred in y_preds:

    probs_pos_XGB.append(pred[1])
roc_gnb = roc_auc_score(y_test,probs_pos_XGB)

print(roc_gnb)
GNB = GaussianNB()
GNB.fit(x_train,y_train)

y_preds_test = GNB.predict_proba(x_test)



probs_pos_test_gnb  = []

for pred in y_preds_test:

    probs_pos_test_gnb.append(pred[1])

    

roc_test = roc_auc_score(y_test,probs_pos_test_gnb)

print(roc_test)
auc = {}

for weight in [x/100 for x in range(0,101)]:

    combined_preds = []

    for i in range(x_test.shape[0]):

        combined_pred = probs_pos_XGB[i] * weight + probs_pos_test_gnb[i] * (1-weight)

        combined_preds.append(combined_pred)

    auc[weight] = roc_auc_score(y_test,combined_preds)

    
optimal_weight = max(auc, key=auc.get)
x_train.shape
x_sub = np.array(test.iloc[:,1:])

y_probs_sub_XGB = XGB.predict_proba(x_sub)

y_probs_sub_GNB = GNB.predict_proba(x_sub)
probs_pos_sub_XGB  = []

for pred in y_probs_sub_XGB:

    probs_pos_sub_XGB.append(pred[1])

print(len(probs_pos_sub_XGB))
probs_pos_sub_GNB  = []

for pred in y_probs_sub_GNB:

    probs_pos_sub_GNB.append(pred[1])

print(len(probs_pos_sub_GNB))
sub_combined_preds = []

for i in range(x_sub.shape[0]):

    combined_pred = probs_pos_sub_XGB[i] * optimal_weight + probs_pos_sub_GNB[i] * (1-optimal_weight)

    sub_combined_preds.append(combined_pred)

print(len(sub_combined_preds))
submission = pd.DataFrame(columns = ['ID_code','Target'])
submission['ID_code']= test['ID_code'] 

submission['Target'] = sub_combined_preds
submission.to_csv('Submission.csv',index=False)