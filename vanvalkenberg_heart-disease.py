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
### work in Progress
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
DataFrame = pd.read_csv('../input/heart-disease-uci/heart.csv')
DataFrame.describe()

## lets plot various stats of data sets
Male = len(DataFrame[DataFrame['sex'] == 1])
FeMale = len(DataFrame[DataFrame['sex'] == 0])

highrisk = len(DataFrame[DataFrame['target'] == 1])
lowrisk = len(DataFrame[DataFrame['target'] == 0])

highriskMale = len(DataFrame[ (DataFrame['sex'] == 1) & (DataFrame['target'] == 1)])
highriskFemale = len(DataFrame[ (DataFrame['sex'] == 0) & (DataFrame['target'] == 1)])
plt.figure(figsize = (10, 10))
plt.bar(['Positives','Negatives', 'Males', 'PostiveMales', 'Females','PositiveFemales'],[highrisk, lowrisk, Male, highriskMale, FeMale, highriskFemale])
plt.title('Distribution of Positives and negatives amongst males and females')
plt.ylabel('number of candidates')
plt.grid(True)
## let find age distribution for heart diesease for both males and females
def AgeDistrib(DataFrame):
    ## defining bins
    labels = ['< 20', '20-30','30-40', '40-50', '50-60', '60-70', '70-80', '80 <']
    values = []
    values.append(len(DataFrame[DataFrame['age'] < 20]))
    values.append(len(DataFrame[(DataFrame['age'] >= 20) & (DataFrame['age'] < 30)]))
    values.append(len(DataFrame[(DataFrame['age'] >= 30) & (DataFrame['age'] < 40)]))
    values.append(len(DataFrame[(DataFrame['age'] >= 40) & (DataFrame['age'] < 50)]))
    values.append(len(DataFrame[(DataFrame['age'] >= 50) & (DataFrame['age'] < 60)]))
    values.append(len(DataFrame[(DataFrame['age'] >= 60) & (DataFrame['age'] < 70)]))
    values.append(len(DataFrame[(DataFrame['age'] >= 70) & (DataFrame['age'] < 80)]))
    values.append(len(DataFrame[(DataFrame['age'] >= 80)]))
    
    return labels, values
mlabels, mValues = AgeDistrib(DataFrame[DataFrame['sex'] == 1])
flabels, fValues = AgeDistrib(DataFrame[DataFrame['sex'] == 0])
plt.figure(figsize = (10, 10))
plt.bar(mlabels, mValues)
plt.title('Heart disease distribution with Age (Males)')
plt.ylabel('number of candidates')
plt.grid(True)
plt.figure(figsize = (10, 10))
plt.bar(flabels, fValues)
plt.title('Heart disease distribution with Age (Females)')
plt.ylabel('number of candidates')
plt.grid(True)
## lets find the correleations
import seaborn as sb
plt.figure(figsize = (10, 10))
sb.heatmap(DataFrame.corr(), annot = True)

## This implies heart disease has strong cporrelation with chestPain, maximumHearRateAchieved, 
## let us now build 2 models one with all features of dataset and one with only those features that posses a strong correlation
FeatureVector1 = DataFrame[DataFrame.columns]
FeatureVector2 = DataFrame[['age', 'sex', 'cp', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
labels = DataFrame['target']
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(np.array(FeatureVector1), np.array(labels), test_size = 0.1 , random_state = 42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(np.array(FeatureVector2), np.array(labels), test_size = 0.1 , random_state = 42)

s1, f1 = X_train1.shape
s2, f2 = X_train2.shape
Mymodel1 = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape = (f1, )),
    tf.keras.layers.Dense(50, activation = 'relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(25, activation = 'relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation = 'relu', kernel_regularizer='l2'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    
])

Mymodel1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
retVal = Mymodel1.fit(X_train1, y_train1, validation_data = (X_test1, y_test1),epochs = 300 )
Mymodel2 = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape = (f2, )),
    tf.keras.layers.Dense(50, activation = 'relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(25, activation = 'relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation = 'relu', kernel_regularizer='l2'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    
])

Mymodel2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
retVal2 = Mymodel2.fit(X_train2, y_train2, validation_data = (X_test2, y_test2),epochs = 300 )
PredByModel1 = Mymodel1.predict(np.array(FeatureVector1)).flatten()
PredByModel1 = np.round_(PredByModel1)
predByModel2 = Mymodel2.predict(np.array(FeatureVector2)).flatten()
PredByModel2 = np.round_(predByModel2)
from sklearn.metrics import accuracy_score
accuracy_score (np.array(labels), PredByModel1)
accuracy_score (np.array(labels), PredByModel2)