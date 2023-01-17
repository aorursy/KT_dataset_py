#Importing the libraries

import numpy as np

import pandas as pd
# Importing the dataset

dataset = pd.read_csv('../input/Kickstarter_projects_Feb19.csv', encoding= 'latin-1')



# Filtering data

# Only including projects that have either failed or successful as their state of success



ds1 = dataset[(dataset.status == 'successful')]

ds2 = dataset[(dataset.status == 'failed')]

ds = [ds1,ds2]

ds3 = pd.concat(ds)



#Splitting the dataset into dependable variable and independent vector

cols = [2,3,4,7,8,9,10,11,12,13]

X = ds3.iloc[:, cols].values

Y = ds3.iloc[:, 14].values

#Y1 = ds4.iloc[:,].values



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1)

#Y_train1,Y_test1 =  train_test_split( Y, test_size = 0.1, random_state = 1)

X.max(axis = 0)





#prepare y train and test

for i in range(0,len(Y_train)):

  if Y_train[i]=="successful":

    Y_train[i]=1.0

  else:

    Y_train[i]=0.0

    

for i in range(0,len(Y_test)):

  if Y_test[i]=="successful":

    Y_test[i]=1.0

  else:

    Y_test[i]=0.0

    

#Catboost

from catboost import CatBoostClassifier

cat_feat = [0,1,2,5,6,7]

model=CatBoostClassifier(learning_rate=0.21)

model.fit(X_train, Y_train,cat_features=cat_feat,logging_level='Silent')



# Predicting the Test set results

y_pred = model.predict(X_test)



#create confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test.astype(int), y_pred.round())



#print accuracy

from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test.astype(int),y_pred.round())*100)