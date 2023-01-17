import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Plot



import os

print(os.listdir("../input"))

# Importing Datasets

train_dataset = pd.read_csv('../input/train.csv')

predict_dataset = pd.read_csv('../input/test.csv')



# Data Set

X = train_dataset.iloc[:, [2,4,5,6,7,9,11]].values

y = train_dataset.iloc[:, 1].values



# Predicting Set

X_predict = predict_dataset.iloc[:, [1,3,4,5,6,8,9]].values



# Splitting data into different sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Taking care of missing data 

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy= 'most_frequent')

imputer = imputer.fit(X_train)

X_train = imputer.transform(X_train)





imputer_test = SimpleImputer(missing_values=np.nan, strategy= 'most_frequent')

imputer_test = imputer_test.fit(X_test)

X_test = imputer.transform(X_test)





imputer_predict = SimpleImputer(missing_values=np.nan, strategy= 'most_frequent')

imputer_predict = imputer_predict.fit(X_predict)

X_predict = imputer_predict.transform(X_predict)

# Label Encoder Encode categorical data 

from sklearn.preprocessing import LabelEncoder

# Gender

labelencoder_gender = LabelEncoder()

X_train[:, 1] = labelencoder_gender.fit_transform(X_train[:, 1])

X_test[:, 1] = labelencoder_gender.fit_transform(X_test[:, 1])

X_predict[:, 1] = labelencoder_gender.fit_transform(X_predict[:, 1])



# Embarked

labelencoder_embarked = LabelEncoder()

X_train[:, 6] = labelencoder_embarked.fit_transform(X_train[:, 6])

X_test[:, 6] = labelencoder_embarked.fit_transform(X_test[:, 6])

X_predict[:, 6] = labelencoder_embarked.fit_transform(X_predict[:, 6])



# Fit RandomForest Classifier

#from sklearn.ensemble import RandomForestClassifier

#classifier = RandomForestClassifier(n_estimators=2000, criterion='entropy', random_state=0)

#classifier.fit(X_train, y_train)

# Fitting the Logistic Regression Model to the dataset

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train,y_train)



# Predictions

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
# Applying k-Fold Cross Validation

#from sklearn.model_selection import cross_val_score

#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

#accuracies.mean()

#accuracies.std()



# Applying Grid Search to find the best model and the best parameters

#from sklearn.model_selection import GridSearchCV

#parameters = [{'n_estimators': [2000, 2100,2200,2500], 'criterion': ['entropy']},

#              {'n_estimators': [2000, 2100,2200,2500], 'criterion': ['gini']}]

#grid_search = GridSearchCV(estimator = classifier,

#                           param_grid = parameters,

#                           scoring = 'accuracy',

#                           cv = 10,

#                           n_jobs = -1)

#grid_search = grid_search.fit(X_train, y_train)

#best_accuracy = grid_search.best_score_

#best_parameters = grid_search.best_params_

# Prediction for Test Data Set

y_test_data_predition = classifier.predict(X_predict)

print(y_test_data_predition)
# Append Result to Test Data Set

predict_dataset['Survived'] = y_test_data_predition

print(predict_dataset.values)

# Extract Expected Data

result_dataset = predict_dataset.iloc[:, [0, 11]]



# Write to New CSV File

result_dataset.to_csv('Result.csv',  index=False) # Remove Index Column