import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt2

import matplotlib.cm as cm

%matplotlib inline

from sklearn import preprocessing

from subprocess import check_output



#

print(check_output(["ls", "../input"]).decode("utf8"))







# Read the data file

data = pd.read_csv('../input/data.csv')

data.head()

# Cleaning and modifying the data

data = data.drop('id',axis=1)

data = data.drop('Unnamed: 32',axis=1)

# Mapping Benign to 0 and Malignant to 1 

data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

# Scaling the dataset

datas = pd.DataFrame(preprocessing.scale(data.iloc[:,1:32]))

datas.columns = list(data.iloc[:,1:32].columns)

datas['diagnosis'] = data['diagnosis']

# Creating the high dimensional feature space X

data_drop = datas.drop('diagnosis',axis=1)

X = data_drop.values



# Create a feed forward neural network with 3 hidden layers

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Input 

from keras.optimizers import SGD



model = Sequential()

model.add(Dense(128,activation="relu",input_dim = np.shape(X)[1]))

model.add(Dropout(0.25))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])



# Fit and test the model by randomly splitting it 

# 67% of the data for training and 33% of the data for validation

model.fit(X, datas['diagnosis'], batch_size=5, epochs=10,validation_split=0.33)





# Cross validation analysis 

from sklearn.model_selection import StratifiedKFold

# fix random seed for reproducibility

seed = 3

np.random.seed(seed)

# K fold cross validation (k=2)

k = 2

kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)

cvscores = []

Y = datas['diagnosis']

for train, test in kfold.split(X, Y):

    # Fit the model

    model.fit(X[train], Y[train], epochs=10, batch_size=10, verbose=0)

    # evaluate the model

    scores = model.evaluate(X[test], Y[test], verbose=0)

    # Print scores from each cross validation run 

    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    cvscores.append(scores[1] * 100)

print("%d-fold cross validation accuracy -  %.2f%% (+/- %.2f%%)" % (k,np.mean(cvscores), np.std(cvscores)))