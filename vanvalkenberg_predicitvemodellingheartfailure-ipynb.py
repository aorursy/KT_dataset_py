##################################################################

#                 Contents                                       #

##################################################################

# 1.) Data Preprocessing                                         #

# 2.) Defining and Training Ann using Tensorflow                 #

# 3.) Training Deccision tree, Random Forest & Gradient Boosting #

#     Classifier                                                 #

# 4.) Conclusions                                                #

##################################################################
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
DataFrame = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

DataFrame.loc[0:9]
DataFrame.info()
import seaborn as sb

import matplotlib.pyplot as plt

import tensorflow as tf
## plotting correlation 

plt.figure(figsize=(20,10))

sb.heatmap(DataFrame.corr(), annot = True)

plt.title('Correlation Matrix ')
## It seems age,  serum_creatinine and follow are time shares a strong correlation with chances of patient dying due to 

## Heart Attack
## let divide our data into training and testing sets 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



FeatureVector = DataFrame[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',

       'ejection_fraction', 'high_blood_pressure', 'platelets',

       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]



labelVector = DataFrame['DEATH_EVENT']



## Converting to numpy arrays and applying StandarScaler to scale our data

FeatureVector = np.array(FeatureVector, dtype = 'float64')

labelVector = np.array(labelVector, dtype = 'float64')



stdScaller = StandardScaler()

FeatureVector = stdScaller.fit_transform(FeatureVector)



## Splitting our data into test set and train sets using 10 % data for testing

X_train, X_test, Y_train, Y_test = train_test_split(FeatureVector, labelVector, random_state = 42, test_size = 0.1 )
samp, features = X_train.shape
## lets first pit this  data to a nueral network

myAnn = tf.keras.models.Sequential([

    tf.keras.layers.Input(shape= (features,)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(500, activation = 'relu',  kernel_regularizer = 'l2'),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(125, activation = 'relu',  kernel_regularizer='l2'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(62, activation = 'relu',  kernel_regularizer='l2'),

    tf.keras.layers.Dense(1, activation = 'sigmoid',  kernel_regularizer='l1')

    

])
myAnn.compile (

optimizer='adam',

metrics = ['accuracy'],

loss = 'binary_crossentropy'

)



myAnn.summary()
retVal = myAnn.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=10, epochs =500)
plt.plot(retVal.history['loss'], label = 'training loss')

plt.plot(retVal.history['val_loss'], label = 'validation loss')

plt.legend()

plt.title('losses')

plt.show()

plt.plot(retVal.history['accuracy'], label = 'valoidation accuracy')

plt.plot(retVal.history['val_accuracy'], label = 'validation accuracy')

plt.legend()

plt.title('Accuracies')

plt.show()
from sklearn.metrics import accuracy_score, confusion_matrix
## lets define a performance function 

def RateMyModelsPerformance(model, name):

    predictions = model.predict(X_test)

    predictions = np.round_(predictions)

    

    print ('Model Name:{}'.format(name))

    print('Model accuracy:{}'.format(accuracy_score(Y_test, predictions)))

    print ('Confussion Matrix:\n{}'.format(confusion_matrix(Y_test, predictions)))

    
RateMyModelsPerformance(myAnn, 'Neural network validation')
## lets Train a decission tree, random Forest and gradient Boosting classifier

from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()

DTC.fit(X_train,Y_train)



RateMyModelsPerformance(DTC, 'Decission Tree Classifier')
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators = 50)

RFC.fit(X_train, Y_train)



RateMyModelsPerformance(RFC, 'Random Forest Classifier')
from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(n_estimators= 100)

GBC.fit(X_train, Y_train)



RateMyModelsPerformance(GBC, 'Gradient Bosting Classifier')
## lets train a SVM Classifier as well

from sklearn.svm import SVC

svc = SVC(kernel='linear')

svc.fit(X_train, Y_train)



RateMyModelsPerformance(svc, 'Support Vector Classifier linear kernel')
#############################################################

#                     Conclusions                           #

#############################################################

# Ann Accuracy  =  70 %                                     #

# Decission Tree accuracy = 70 %                            #

# Random Forest accurayc =  73.333%                         #

# gradient Bosting Classifier accuracay = 76.67%            #

# Support Vector Classifier linear kernel accuracy = 80%    #

#############################################################