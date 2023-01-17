# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#author: trhimtu
#packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#dataset

dataset = pd.read_csv('../input/mushrooms.csv')


#explore dataset

    #  dataset.head(10)
    #  dataset.describe()
    #  dataset.info()
    #  dataset.loc[dataset["EDIBLE"].isnull()]

#assign X and y
X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]].values
y = dataset.iloc[:, [0]].values
#X = pd.DataFrame(X)
#y = pd.DataFrame(y)

# Encoding categorical data for X dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_0 = LabelEncoder()
X[:, 0] = labelencoder_X_2.fit_transform(X[:, 0])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_2.fit_transform(X[:, 4])
labelencoder_X_5 = LabelEncoder()
X[:, 5] = labelencoder_X_2.fit_transform(X[:, 5])
labelencoder_X_6 = LabelEncoder()
X[:, 6] = labelencoder_X_2.fit_transform(X[:, 6])
labelencoder_X_7 = LabelEncoder()
X[:, 7] = labelencoder_X_2.fit_transform(X[:, 7])
labelencoder_X_8 = LabelEncoder()
X[:, 8] = labelencoder_X_2.fit_transform(X[:, 8])
labelencoder_X_9 = LabelEncoder()
X[:, 9] = labelencoder_X_2.fit_transform(X[:, 9])
labelencoder_X_10 = LabelEncoder()
X[:, 10] = labelencoder_X_2.fit_transform(X[:, 10])
labelencoder_X_11 = LabelEncoder()
X[:, 11] = labelencoder_X_2.fit_transform(X[:, 11])
labelencoder_X_12 = LabelEncoder()
X[:, 12] = labelencoder_X_2.fit_transform(X[:, 12])
labelencoder_X_13 = LabelEncoder()
X[:, 13] = labelencoder_X_2.fit_transform(X[:, 13])
labelencoder_X_14 = LabelEncoder()
X[:, 14] = labelencoder_X_2.fit_transform(X[:, 14])
labelencoder_X_15 = LabelEncoder()
X[:, 15] = labelencoder_X_2.fit_transform(X[:, 15])
labelencoder_X_16 = LabelEncoder()
X[:, 16] = labelencoder_X_2.fit_transform(X[:, 16])
labelencoder_X_17 = LabelEncoder()
X[:, 17] = labelencoder_X_2.fit_transform(X[:, 17])
labelencoder_X_18 = LabelEncoder()
X[:, 18] = labelencoder_X_2.fit_transform(X[:, 18])
labelencoder_X_19 = LabelEncoder()
X[:, 19] = labelencoder_X_2.fit_transform(X[:, 19])
labelencoder_X_20 = LabelEncoder()
X[:, 20] = labelencoder_X_2.fit_transform(X[:, 20])
labelencoder_X_21 = LabelEncoder()
X[:, 21] = labelencoder_X_2.fit_transform(X[:, 21])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
#encoding for y
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_0 = LabelEncoder()
y[:, 0] = labelencoder_y_0.fit_transform(y[:, 0])

#split data -try the model with random state 42 and also it is chwcked with random state 0
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#deep learning with keras lib
import keras
from keras.models import Sequential
from keras.layers import Dense
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu', input_dim = 24))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))
# Adding the third hidden layer
#classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set-I have tried it with different batch sizes and epochs but it was overfitted
classifier.fit(X_train, y_train, batch_size = 100, nb_epoch = 7)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = y_pred.round().astype(int)
y_test = y_test.astype(int)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

