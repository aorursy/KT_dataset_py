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
import seaborn as sb

import matplotlib.pyplot as plt
DataFrame = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

DataFrame.info()
## Correlation matrix

sb.heatmap(DataFrame.corr(), annot = True)

## glucose, bmi, age, pregnancies, insulin, diabetesePedigree has high correlation with diabetes
g20_30 = len (DataFrame[(DataFrame['Age'] >= 20) & (DataFrame['Age'] < 30) & (DataFrame['Outcome'] == 1)])

g30_40 = len (DataFrame[(DataFrame['Age'] >= 30) & (DataFrame['Age'] < 40) & (DataFrame['Outcome'] == 1)])

g40_50 = len (DataFrame[(DataFrame['Age'] >= 40) & (DataFrame['Age'] < 50) & (DataFrame['Outcome'] == 1)])

g50_60 = len (DataFrame[(DataFrame['Age'] >= 50) & (DataFrame['Age'] < 60) & (DataFrame['Outcome'] == 1)])

g60_70 = len (DataFrame[(DataFrame['Age'] >= 60) & (DataFrame['Age'] < 70) & (DataFrame['Outcome'] == 1)])

g70_80 = len (DataFrame[(DataFrame['Age'] >= 70) & (DataFrame['Age'] < 80) & (DataFrame['Outcome'] == 1)])

g80_above =len (DataFrame[(DataFrame['Age'] >= 80)  & (DataFrame['Outcome'] == 1)])
plt.bar(['20-30', '30-40', '40-50', '50-60', '60-70','70-80', '80 above'],[g20_30, g30_40, g40_50,g50_60,g60_70,g70_80,g80_above])

plt.xlabel('Age')

plt.ylabel('number of positive patients')

plt.title('Diabetes positive patients age distribution')

plt.grid(True)

plt.legend()
## lets try to classify positive patients on basis BMI

## BMI has the following categories

## underwiegth  < 18.5

## normal  18.5 -24.9

## Overweight = 25 - 29.9

## obese > 30



underweight = len (DataFrame[(DataFrame['BMI'] <= 18.5)  & (DataFrame['Outcome'] == 1)])

normal  = len (DataFrame[(DataFrame['BMI'] > 18.5 ) & (DataFrame['BMI'] <= 24.9) & (DataFrame['Outcome'] == 1)])

overwieht = len (DataFrame[(DataFrame['BMI'] > 24.9 ) & (DataFrame['BMI'] <= 29.9) & (DataFrame['Outcome'] == 1)])

obese = len (DataFrame[(DataFrame['BMI'] > 29.9)  & (DataFrame['Outcome'] == 1)])



plt.bar(['under weight', 'normal weigth', 'Over weigth', 'obese'],[underweight, normal, overwieht, obese])

plt.xlabel('BMI')

plt.ylabel('number of positive patient')

plt.title('Diabetes positive patients BMI distribution')

plt.grid(True)

plt.legend()
## let train a neural network for predicting diabetes

import tensorflow as tf

import sklearn 

from sklearn.model_selection import train_test_split
Features = np.array (DataFrame[['Pregnancies','Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] )

label = np.array (DataFrame['Outcome'])
X_train, X_test, y_train, y_test = train_test_split(Features,label, test_size = 0.1, random_state = 42)

sample, featureLenght = Features.shape

# model

Mymodel = tf.keras.models.Sequential([

    

    tf.keras.layers.Input(shape = (featureLenght,)),

    tf.keras.layers.Dense(30, activation = 'relu', kernel_regularizer='l2'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(15, activation = 'relu', kernel_regularizer='l2'),

    #tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(1, activation = 'sigmoid')

])



Mymodel.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])



retVal = Mymodel.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 300, batch_size = 50)
plt.plot(retVal.history['loss'], label = 'training set loss')

plt.plot(retVal.history['val_loss'], label = 'validation set loss')





plt.legend()
plt.plot(retVal.history['accuracy'], label = 'training set accuracy')

plt.plot(retVal.history['val_accuracy'], label = 'validation set accuracy')



plt.legend()
idx = []

for i in range (len (DataFrame)):

    idx.append(i)



Pred = Mymodel.predict(Features)

Pred = np.round_(Pred)

Pred = Pred.flatten()



from sklearn.metrics import confusion_matrix

confusion_matrix(label, Pred)
