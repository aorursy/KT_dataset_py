import pandas as pd
import numpy as np
import seaborn as sns

%matplotlib inline
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

from sklearn.preprocessing import LabelEncoder

import tensorflow as tf 
import keras 
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation, Dropout
from keras.optimizers import Adam
#import data
data = pd.read_csv('../input/data.csv')
data.head(10)

data['diagnosis'].value_counts()
#Correlation map
corrmat = data.corr()
plt.subplots(figsize=(20,15))
sns.heatmap(corrmat, vmax=0.9, square=True)
#let us check if we have any null values
[col for col in data.columns if data[col].isnull().any()]
#Let us drop 'id' which is just a number not related to 'diagnosis'
#and 'Unnamed: 32' - always takes only nan value and hence we can safely drop it
data=data.drop(['id','Unnamed: 32'], axis = 1)
data.head()
#let us choose only uncorrelated features
#'radius_mean','perimeter_mean','area_mean', 'radius_worst', 'perimeter_worst','area_worst' are correlated - take only 'radius_mean'
#'texture_mean','texture_worst' are correlated - take only 'texture_mean'
#'smoothness_mean','smoothness_worst' correlated-take only 'smoothness_mean'
#'compactness_mean','concavity_mean','concave points_mean','compactness_worst','concavity_worst','concave points_worst', take 'concavity_mean'
#'radius_se','perimeter_se', 'area_se'-take 'radius_se'
# 'compactness_se','concavity_se','concave points_se' take 'concavity_se','

drop_features=['perimeter_mean','area_mean', 'radius_worst', 'perimeter_worst','area_worst','texture_worst','smoothness_worst','perimeter_se', 'area_se','compactness_mean','concave points_mean','compactness_worst','concavity_worst','concave points_worst','compactness_se','concave points_se']
data=data.drop(drop_features, axis = 1)
data.head()


#Correlation map to see how features are correlated with SalePrice
corrmat = data.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corrmat, vmax=0.9, square=True)
#let us randomly shuffle rows in our data (for fair train test splitting)
data.iloc[np.random.permutation(len(data))]

#and now lets split it into features and targets 
Y_data = data['diagnosis']
X_data = data.drop('diagnosis',axis=1)

#Let us change 'M' and 'B' in 'diagnosis' column to numerical values
le = LabelEncoder()
Y_data = le.fit_transform(Y_data)

#and split train and test data
X_train,X_test,Y_train,Y_test = train_test_split(X_data,Y_data,test_size = 0.2,random_state = 1)
# LogisticRegression - gives the best results
logreg = LogisticRegression(C=5000, solver='newton-cg', multi_class='multinomial',max_iter=10000)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

print ("score: %s" % logreg.score(X_train, Y_train)) 
print ("accuracy score: %s" % accuracy_score(Y_test, Y_pred))
print ("precision score: %s" % precision_score(Y_test, Y_pred))
print ("recall score: %s" % recall_score(Y_test, Y_pred))
print ("f1 score: %s" %  f1_score(Y_test, Y_pred))

cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm,annot=True)
# Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=50, criterion='entropy',random_state=30)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)

print ("score: %s" % random_forest.score(X_train, Y_train)) 
print ("accuracy score: %s" % accuracy_score(Y_test, Y_pred))
print ("precision score: %s" % precision_score(Y_test, Y_pred))
print ("recall score: %s" % recall_score(Y_test, Y_pred))
print ("f1 score: %s" %  f1_score(Y_test, Y_pred))

cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm,annot=True)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#neural network

model = Sequential()

model.add(Dense(units = 16,activation = 'relu',kernel_initializer='uniform', input_dim = 14))

# hidden layer
model.add(Dense(units = 16, activation = 'relu',kernel_initializer='uniform'))
# hidden layer
model.add(Dense(units = 1, activation = 'sigmoid',kernel_initializer='uniform'))
#model.add(Activation(tf.nn.softmax))

# Compiling Neural Network
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting 
model.fit(X_train, Y_train, batch_size = 150, epochs = 200,verbose=2)

# Predicting the Test set results
Y_pred = model.predict_classes(X_test)




print ("accuracy score: %s" % accuracy_score(Y_test, Y_pred))
print ("precision score: %s" % precision_score(Y_test, Y_pred))
print ("recall score: %s" % recall_score(Y_test, Y_pred))
print ("f1 score: %s" %  f1_score(Y_test, Y_pred))

cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm,annot=True)

