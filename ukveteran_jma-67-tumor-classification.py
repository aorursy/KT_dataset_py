# Load the libraries

import numpy as np

import pandas as pd



# Import the dataset as a pandas dataframe object

dataset = pd.read_csv("../input/deeplearning-wisconsin-breastcancer.csv")



# drop extra empty column

dataset = dataset.drop('Unnamed: 32', axis = 1)



# change 'B' and 'M' (Benign and Malignant) to zeros and ones (the class labels)

dataset.diagnosis.replace(['B', 'M'], [0, 1], inplace=True)



# create X and y numpy arrays for input into the models

X = dataset.iloc[:, 2:].values

y = dataset.iloc[:, 1].values



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)



# Splitting the dataset into the Training set and Testing set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 65)





# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout



# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 16, activation = 'relu', input_dim = 30))

classifier.add(Dropout(rate = 0.1))



# Adding the second hidden layer

classifier.add(Dense(units = 16, activation = 'relu'))

classifier.add(Dropout(rate = 0.1))



# Adding the output layer

classifier.add(Dense(units = 1, activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)



# Making predictions and evaluating the model



# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



# Measuring the model accuracy

accuracy = np.trace(cm) / np.sum(cm)

precision = cm[1][1] / (cm[0][1] + cm[1][1])

recall = cm[1][1] / (cm[1][0] + cm[1][1])

print('Accuracy: \t', accuracy)

print('Precision: \t', precision)

print('Recall: \t', recall)







# Evaluating, Improving and Tuning the ANN



# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense

def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))

    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 6)

print("Accuracy: ", accuracies.mean())

print("SD: ", accuracies.std())



# Improving the ANN

# add dropout regularization to reduce overfitting (see above)



# Tuning the ANN (grid search pf parameters)

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential

from keras.layers import Dense

def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units = 16, activation = 'relu', input_dim = 30))

    classifier.add(Dense(units = 16, activation = 'relu'))

    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [10, 15, 20, 30],

              'epochs': [50, 75, 100, 150],

              }

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 6)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_