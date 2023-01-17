# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

myStop = 0

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        myStop += 1

        print(os.path.join(dirname, filename))

        if myStop==20:

            break

        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image

from tqdm import tqdm

import glob

import gc

import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV

from random import random

from sklearn import metrics
import scipy.stats as dist
train_df = pd.read_csv("/kaggle/input/garage-detection-unofficial-ssl-challenge/image_labels_train.csv")

train_df.head()
Image.open("/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image1607.jpg")
np.array(Image.open("/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image1607.jpg")).shape
np.array(Image.open("/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image1607.jpg").resize((224, 224)))[:,:,0].flatten().shape
(224, )*2
224*224
def preprocess_image(image_path, desired_size=224):

    im = Image.open(image_path)

    im = im.resize((desired_size, )*2, resample=Image.LANCZOS)

    im = np.array(im)

    if len(im.shape)==3:

        im = im[:,:,0]

    im = im.flatten()

    return im
# get the number of training images from the target\id dataset

N = train_df.shape[0]

# create an empty matrix for storing the images

x_train = np.empty((N, 50176), dtype=np.uint8)



# loop through the images from the images ids from the target\id dataset

# then grab the cooresponding image from disk, pre-process, and store in matrix in memory

for i, image_id in enumerate(tqdm(train_df['ID'])):

    x_train[i, :] = preprocess_image(

        f'/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image{image_id}.jpg'

    )
holdout_df = pd.read_csv("/kaggle/input/garage-detection-unofficial-ssl-challenge/image_labels_holdout.csv")

holdout_df.head()
# get the number of training images from the target\id dataset

N = holdout_df.shape[0]

# create an empty matrix for storing the images

x_holdout = np.empty((N, 50176), dtype=np.uint8)



# loop through the images from the images ids from the target\id dataset

# then grab the cooresponding image from disk, pre-process, and store in matrix in memory

for i, image_id in enumerate(tqdm(holdout_df['ID'])):

    x_holdout[i, :] = preprocess_image(

        f'/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image{image_id}.jpg'

    )
unlabeledIDs = []

labeledIDs = holdout_df['ID'].tolist() + train_df['ID'].tolist()

for file in tqdm(glob.glob('/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/*.jpg')):

    myStart = file.find('/image')

    myEnd = file.find('.jpg')

    myID = file[myStart+6:myEnd]

    if int(myID) not in labeledIDs:

        unlabeledIDs.append(myID)
# get the number of training images from the target\id dataset

N = len(unlabeledIDs)

# create an empty matrix for storing the images

x_unlabeled = np.empty((N, 50176), dtype=np.uint8)



# loop through the images from the images ids from the target\id dataset

# then grab the cooresponding image from disk, pre-process, and store in matrix in memory

for i, image_id in enumerate(tqdm(unlabeledIDs)):

    x_unlabeled[i, :] = preprocess_image(

        f'/kaggle/input/garage-detection-unofficial-ssl-challenge/GarageImages/GarageImages/image{image_id}.jpg'

    )
x_train, x_test, y_train, y_test = train_test_split(x_train, 

                                                    train_df['GarageDoorEntranceIndicator'], 

                                                    test_size=0.50, 

                                                    random_state=42, 

                                                    stratify=train_df.GarageDoorEntranceIndicator)
print("train size: ", len(y_train))

print("train total 1s: ", sum(y_train))

print("test size: ", len(y_test))

print("test total 1s: ", sum(y_test))
# distributions = {

#     'n_estimators': dist.randint(1,500),

#     'criterion': ['gini','entropy'],

#     'max_depth': dist.randint(1,20),

#     'min_samples_split': dist.uniform(0.1,0.9),

#     'min_samples_leaf': dist.uniform(0,0.5),

#     'min_weight_fraction_leaf': dist.uniform(0,0.5),

#     'max_features': dist.uniform(0.1,0.9),

#     'min_impurity_decrease': dist.uniform(0,1),

#     'class_weight': ['balanced','balanced_subsample'],

#     'max_samples': dist.uniform(0.1,0.9)

# }
# %%time



# clf = RandomForestClassifier(n_jobs=-1, verbose=0)

# randGrid = RandomizedSearchCV(clf, distributions, n_iter=40, cv=2, random_state=1, verbose=2, n_jobs=-1)

# randGrid.fit(x_train, y_train)

# print("Best score was {} using {}".format(randGrid.best_score_,randGrid.best_params_))
def get_model():

    # Best score was 0.6896551724137931 using params..

    params = {

        'class_weight': 'balanced', 

        'criterion': 'entropy', 

        'max_depth': 11, 

        'max_features': 0.42473327898450497,

        'max_samples': 0.3471806361002021, 

        'min_impurity_decrease': 0.07396899351720321, 

        'min_samples_leaf': 0.07606857919628118, 

        'min_samples_split': 0.24545667228018142, 

        'min_weight_fraction_leaf': 0.4693558498522599, 

        'n_estimators': 392

    }

    model = RandomForestClassifier(n_jobs=-1, verbose=0, n_estimators=params['n_estimators'], criterion=params['criterion'], max_depth=params['max_depth'],

                                   min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'],

                                   min_weight_fraction_leaf=params['min_weight_fraction_leaf'], max_features=params['max_features'], 

                                   min_impurity_decrease=params['min_impurity_decrease'], class_weight=params['class_weight'], 

                                   max_samples=params['max_samples'], random_state=1)

    return model
def get_auc(X,Y):

    probabilityOf1 = model.predict_proba(X)[:,1]

    fpr, tpr, thresholds = metrics.roc_curve(Y, probabilityOf1, pos_label=1)

    return metrics.auc(fpr, tpr)
%%time



sslRounds = 4

x_train_ssl = np.concatenate((x_train, x_unlabeled), axis=0)

testAUCs = []

for sslRound in range(sslRounds):

    # define model

    #clf = RandomForestClassifier(n_jobs=-1, verbose=0)

    #randGrid = RandomizedSearchCV(clf, distributions, n_iter=5, cv=2, random_state=None, verbose=0, n_jobs=-1)

    model = get_model()

    #model = RandomForestClassifier(n_jobs=-1, verbose=0, n_estimators=80, max_depth=1)

    # fit

    if sslRound==0:

        # first round, fit on just labeled data

        model.fit(x_train, y_train)

    else:

        # all other rounds, fit on sample of data

        #x_train_ssl_sample, _, y_train_ssl_sample, _ = train_test_split(x_train_ssl, y_train_ssl, test_size=0.80, random_state=None, stratify=y_train_ssl)

        model.fit(x_train_ssl, y_train_ssl)

    # extract best model

    #model = randGrid.best_estimator_

    # score unlabeled data

    predictions = model.predict_proba(x_unlabeled)[:,1]

    # set random threshold

    #threshold = min(max(0.33,random()),0.37)

    threshold = 0.34

    # print("threshold selected: ", threshold)

    # create pseudo lables based on threshold

    pseudoLabels = np.where(predictions>threshold,1,0)

    # add pseudo labels to next round of training 

    y_train_ssl = np.concatenate((y_train, pseudoLabels), axis=0)

    # get performance metrics

    testAUC = get_auc(x_test,y_test)

    testAUCs.append(testAUC)

    # print performance on test

    print("round {} test auc: {}".format(sslRound,testAUC))

    # clean up

    if sslRound<(sslRounds-1):

        del model

        gc.collect()
histdf = pd.DataFrame()

histdf['test auc'] = testAUCs

histdf[['test auc']].plot()
holdoutPreds = model.predict_proba(x_holdout)[:,1] 

fpr, tpr, thresholds = metrics.roc_curve(holdout_df['GarageDoorEntranceIndicator'], holdoutPreds, pos_label=1)

print("final holdout auc: ", metrics.auc(fpr, tpr))
del model

gc.collect()