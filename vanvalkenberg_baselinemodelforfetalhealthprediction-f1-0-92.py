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
DataFrame = pd.read_csv('/kaggle/input/fetal-health-classification/fetal_health.csv')

DataFrame.info()
## Plotting a histogram of Dataframe's Param values

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (30,30))

ax = fig.gca()

DataFrame.hist(ax = ax)
## let plot correlation 

plt.figure(figsize=(20,20))

import seaborn as sb

sb.heatmap(DataFrame.corr(),annot = True, cmap='coolwarm')
import tensorflow as tf2

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



FeatureVector = np.array(DataFrame[['baseline value', 'accelerations', 'fetal_movement',

       'uterine_contractions', 'light_decelerations', 'severe_decelerations',

       'prolongued_decelerations', 'abnormal_short_term_variability',

       'mean_value_of_short_term_variability',

       'percentage_of_time_with_abnormal_long_term_variability',

       'mean_value_of_long_term_variability', 'histogram_width',

       'histogram_min', 'histogram_max', 'histogram_number_of_peaks',

       'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',

       'histogram_median', 'histogram_variance', 'histogram_tendency']])



LabelVector = np.array(DataFrame['fetal_health'])



X_train, X_test, y_train, y_test = train_test_split(FeatureVector, LabelVector, test_size = 0.1, random_state = 42)



std = StandardScaler()

X_train = std.fit_transform(X_train)

X_test = std.transform(X_test)



rows, feat = X_train.shape
### Defining tensorflow model

MyModel = tf2.keras.models.Sequential([

    tf2.keras.layers.Input(shape= (feat, ),),

    tf2.keras.layers.BatchNormalization(),

    tf2.keras.layers.Dense(500, activation = 'relu', kernel_regularizer='l2'),

    tf2.keras.layers.Dropout(0.2),

    tf2.keras.layers.Dense(300,activation = 'relu', kernel_regularizer='l2' ),

    tf2.keras.layers.Dropout(0.1),

    tf2.keras.layers.Dense(100,activation = 'relu'),

    tf2.keras.layers.Dense(3, activation = 'softmax')

])

############################################

#         Some Adjustments Required        #

############################################

y_train = y_train - 1

y_test = y_test - 1
MyModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'] )

retVal = MyModel.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size= 100, epochs= 500)
## Plotting loss

plt.plot(retVal.history['loss'], label = 'training loss')

plt.plot(retVal.history['val_loss'], label = 'validation loss')

plt.legend()

plt.grid(True)
## Plotting accuracy 

plt.plot(retVal.history['accuracy'], label = 'training accuracy')

plt.plot(retVal.history['val_accuracy'], label = 'validation accuracy')

plt.legend()

plt.grid(True)
from sklearn.metrics import f1_score

Predictions = MyModel.predict(X_test)

AnsArr = []

for mem in Predictions:

    AnsArr.append(np.argmax(mem))
AnsArr = np.array(AnsArr).reshape(-1,)
print('F1_score : {}'.format(f1_score(y_test, AnsArr, average = 'micro')))
