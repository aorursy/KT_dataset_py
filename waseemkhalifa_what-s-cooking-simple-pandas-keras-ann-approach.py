# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Library Loads
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras import backend
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.metrics import accuracy_score
from keras.layers import Dense,Embedding,Flatten,Dropout
from keras.losses import categorical_crossentropy
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import RMSprop
import re
from numpy import argmax
raw_training = pd.read_json("../input/train.json")
raw_test = pd.read_json("../input/test.json")

training_set = raw_training.copy()
test_set = raw_test.copy()
# lets look the at the first 5 rows
training_set.head()
test_set.head()
# missing the cuisine column - the variable we are trying to predict (classification problem)
# change the order of the columns to how I like it!
training_set = training_set[["id","cuisine","ingredients"]]
training_set.head()
training_set.info()
# we have no missing values - Good!
# countplot of cuisine
f, ax = plt.subplots(figsize = (18, 4))
sns.countplot(training_set["cuisine"])
plt.show()

# we have 20 different cuisines
# lets combine training_set & test_set, so all data manipulations don't have to be repeated
training_set["training"] = 1
test_set["training"] = 0
test_set["cuisine"] = "test"
test_set = test_set[["id", "training", "cuisine", "ingredients"]]
training_set = training_set[["id", "training", "cuisine", "ingredients"]]
training_set = pd.concat([training_set, test_set], axis=0)
# de-list the ingredients column and have each ingredient as it's own column in a helper dataframe called ingredients
ingredients = training_set["ingredients"].apply(pd.Series)
ingredients.head()
# now merge with original & delete old ingredients column
training_set = pd.concat([training_set, ingredients], axis=1)
training_set = training_set.drop(columns = ["ingredients"])
training_set.head()
# now transform all the ingredients into one row each by each id and cuisine its attached to
training_set = training_set.melt(id_vars = ["id", "training", "cuisine"], value_name = "ingredient")
training_set = training_set.drop(columns = ["variable"])
training_set.dropna(subset=["ingredient"], inplace = True)
training_set.head()
# lets do some modifications to the ingredients column
# Clean the ingreidents columns
# make all words lowers case
# remove non alphabetic symbols (e.g numbers and symbols like '-')
# remove unit measurements of ingreidients
training_set["ingredient"] = training_set["ingredient"].astype(str).str.lower()
training_set["ingredient"] = training_set["ingredient"].apply(lambda x: re.sub("[^a-zA-Z]"," ",x))
training_set["ingredient"] = training_set["ingredient"].apply(lambda x: \
                                            re.sub((r'\b(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\b')," ",x))
# remove cuisine column to make pivot of ingredients with count
training_set = training_set.drop(columns = ["cuisine"])
# make training_set into pivot table
# one column for each ingredient (1 if the reciepe has the ingredient, 0 if it did not)
training_set["count"] = 1
training_set = pd.pivot_table(training_set, index = ["id","training"], columns = ["ingredient"], values = "count", \
                           aggfunc = np.sum, fill_value = 0).reset_index().rename_axis(None, axis=1);
training_set.head()
# split out into test & training sets again
test_set = training_set[training_set["training"] == 0].drop(columns = ["training"])
training_set = training_set[training_set["training"] == 1].drop(columns = ["training"])
# merge with raw datasets to ensure ids index remain same and add on cuisine column
training_set = pd.merge(raw_training[["id", "cuisine"]], training_set, how="left",on="id")
test_set = pd.merge(raw_test, test_set, how="left", on="id")
test_set = test_set.drop(columns = ["ingredients"])
# one hot encode the cusine column
le = LabelEncoder()
training_set["cuisine"] = le.fit_transform(training_set["cuisine"])
training_set.head()
test_set.head()
training_set.shape
test_set.shape
test_set = test_set.drop(columns = ["id"]).values
X = training_set.drop(columns = ["id", "cuisine"]).values
y = training_set[["cuisine"]].values
# one hot encode y
onehot_encoder = OneHotEncoder(sparse=False)
y = y.reshape(len(y), 1)
y = onehot_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 0)
X.shape
# Let's test our model on our training set first

# Initialing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2000, kernel_initializer = "uniform",
                     activation = "relu", input_dim = 7120))

# Adding the second hidden layer
classifier.add(Dense(units = 500, kernel_initializer = "uniform",
                     activation = "relu"))

# Add drop out to reduce overfitting
classifier.add(Dropout(0.3))

# Adding the third hidden layer
classifier.add(Dense(units = 250, kernel_initializer = "uniform",
                     activation = "relu"))

# Adding the fourth hidden layer
classifier.add(Dense(units = 125, kernel_initializer = "uniform",
                     activation = "relu"))

# Add drop out to reduce overfitting
classifier.add(Dropout(0.3))

# Adding the fifth hidden layer
classifier.add(Dense(units = 50, kernel_initializer = "uniform",
                     activation = "relu"))

# Adding the output layer
classifier.add(Dense(units = 20, kernel_initializer = "uniform",
                     activation = "softmax"))

# Compiling the ANN
classifier.compile(optimizer = RMSprop(lr=0.0005), loss = "categorical_crossentropy",
                   metrics = ["accuracy"])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 50, epochs = 10)
# let's see how are model fares against our training test set

#predict on training test set
y_pred = classifier.predict(X_test)
y_pred = [np.argmax(i) for i in y_pred]

y_test = [np.argmax(i) for i in y_test]

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# accuracy of 74% - we should improve our data manipulations in the ingredients column before feeding it into our model
# Let's fit model on our full training set 

# Initialing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2000, kernel_initializer = "uniform",
                     activation = "relu", input_dim = 7120))

# Adding the second hidden layer
classifier.add(Dense(units = 500, kernel_initializer = "uniform",
                     activation = "relu"))

# Add drop out to reduce overfitting
classifier.add(Dropout(0.3))

# Adding the third hidden layer
classifier.add(Dense(units = 250, kernel_initializer = "uniform",
                     activation = "relu"))

# Adding the fourth hidden layer
classifier.add(Dense(units = 125, kernel_initializer = "uniform",
                     activation = "relu"))

# Add drop out to reduce overfitting
classifier.add(Dropout(0.3))

# Adding the fifth hidden layer
classifier.add(Dense(units = 50, kernel_initializer = "uniform",
                     activation = "relu"))

# Adding the output layer
classifier.add(Dense(units = 20, kernel_initializer = "uniform",
                     activation = "softmax"))

# Compiling the ANN
classifier.compile(optimizer = RMSprop(lr=0.0005), loss = "categorical_crossentropy",
                   metrics = ["accuracy"])

# Fitting the ANN to the Training set
classifier.fit(X, y, batch_size = 50, epochs = 10)
# fit model to training set

#predict on training test set
y_pred = classifier.predict(test_set)
y_pred = le.inverse_transform([np.argmax(i) for i in y_pred])

ids = raw_test.iloc[:, 0:1].values
ids = ids.flatten()

submission = pd.DataFrame(
        {"id": ids,
         "cuisine": y_pred})
submission.head()
submission.to_csv("submission.csv", index = False)
