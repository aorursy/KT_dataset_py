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
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from keras.metrics import Accuracy
data = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv", engine='python')

y = data['quality']
data.drop(['quality'], inplace=True, axis=1)
data.head()

plt.figure(figsize=(10,10))
sns.heatmap(data.corr())

citricacid = data['fixed acidity'] * data['citric acid']
citric_acidity = pd.DataFrame(citricacid, columns=['citric_accidity'])

density_acidity = data['fixed acidity'] * data['density']
density_acidity = pd.DataFrame(density_acidity, columns=['density_acidity'])


datafinal = data.join(citric_acidity).join(density_acidity)


bins = (2, 6, 8)
gnames = ['bad', 'nice']
y = pd.cut(y, bins = bins, labels = gnames)
enc = LabelEncoder()
yenc = enc.fit_transform(y)
xtrain, xtest, ytrain, ytest = train_test_split(datafinal, yenc, train_size=0.7, test_size=0.3)

scale = StandardScaler()

scaledtrain = scale.fit_transform(xtrain)
scaledtest = scale.transform(xtest)
NeuralModel = Sequential([
                          Dense(128, activation='relu', input_shape=(13,)),
                          Dense(32, activation='relu'),
                          Dense(64, activation='relu'),
                          Dense(64, activation='relu'),
                          Dense(64, activation='relu'),
                          Dense(1, activation='sigmoid')
])
rms = Adam(lr=0.0003)

NeuralModel.compile(optimizer=rms, loss='binary_crossentropy', metrics=['accuracy'])
hist = NeuralModel.fit(scaledtrain, ytrain, epochs=100, validation_data=(scaledtest,ytest))
