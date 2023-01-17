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
## loading dataFrames

DataFrame = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')

DataFrame2 = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
## combining dataFrames

DataFrame3 = pd.concat([DataFrame, DataFrame2])
import matplotlib.pyplot as plt

import seaborn as sb



plt.figure(figsize=(20,7))

sb.heatmap(DataFrame3.corr(), annot = True)

plt.title('Correlation of Datasets Params ')

plt.show()
# From Here we Can easily See that Chance of Admission is postively correlated to the following Factors (Decreasing order):

# 1.) CGPA

# 2.) GRE SCORE

# 3.) TOEFL SCORE

# 4.) University Rating

# 5.) SOP

# 6.) LOR

# 5.) Reasearch
## lets split our data into train and test set and train a Neural Network 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



FeatureVector = DataFrame3[[ 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP','LOR ', 'CGPA', 'Research']]

labelVector = DataFrame3['Chance of Admit ']



FeatureVector = np.array(FeatureVector, dtype = 'float64')

labelVector = np.array(labelVector, dtype = 'float64')



StdSc = StandardScaler()

FeatureVector = StdSc.fit_transform(FeatureVector)



X_train, X_test, Y_train, Y_test = train_test_split(FeatureVector, labelVector, random_state = 42, test_size = 0.1)

import tensorflow as tf
## getting shape of feature vector

num, features = X_train.shape
## Defining our Model

myAnn = tf.keras.models.Sequential([

    tf.keras.layers.Input(shape = (features,)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(256, activation = 'relu', kernel_regularizer='l2'),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128, activation = 'relu', kernel_regularizer='l2'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(64,activation = 'relu', kernel_regularizer='l2'),

    tf.keras.layers.Dense(1,)

])



myAnn.compile(optimizer='adam', loss = 'mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
retVal = myAnn.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size= 100, epochs = 500);
plt.plot(retVal.history['loss'], label= 'training loss')

plt.plot(retVal.history['val_loss'], label= 'validation loss')

plt.legend()

## lets plot predicted and actual probalities 



plt.scatter(Y_test, myAnn.predict(X_test))

plt.plot(np.arange(0.0,1,0.1),np.arange(0.0,1,0.1), color = 'green')

plt.plot(0.5 * np.ones(10), np.arange(0,1,0.1), color = 'yellow')

plt.plot(np.arange(0,1,0.1),0.5 * np.ones(10), color = 'yellow')

plt.xlabel('true probabilities of admission')

plt.ylabel('predicted probalities of admission')

plt.grid(True)

plt.show()