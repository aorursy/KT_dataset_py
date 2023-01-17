import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from collections import Counter

from IPython.core.display import display, HTML

sns.set_style('darkgrid')
dataset = pd.read_csv('/kaggle/input/glass/glass.csv')

dataset.head()
corr = dataset.corr()

#Plot figsize

fig, ax = plt.subplots(figsize=(10, 8))

#Generate Heat Map, allow annotations and place floats in map

sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")

#Apply xticks

plt.xticks(range(len(corr.columns)), corr.columns);

#Apply yticks

plt.yticks(range(len(corr.columns)), corr.columns)

#show plot

plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['RI'], ax = axes[0])

axes[0].set_xlabel('Refractive Index', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.violinplot(x = 'Type', y = 'RI', data = dataset, hue = 'Type', dodge = False, ax = axes[1])

axes[1].set_xlabel('Type of glass', fontsize=14)

axes[1].set_ylabel('Refractive Index', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['Na'], ax = axes[0])

axes[0].set_xlabel('Sodium', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.boxplot(x = 'Type', y = 'Na', data = dataset, hue = 'Type', dodge = False, ax = axes[1])

axes[1].set_xlabel('Type of glass', fontsize=14)

axes[1].set_ylabel('Sodium', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['Mg'], ax = axes[0])

axes[0].set_xlabel('Magnesium', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.violinplot(x = 'Type', y = 'Mg', data = dataset, hue = 'Type', dodge = False, ax = axes[1])

axes[1].set_xlabel('Type of glass', fontsize=14)

axes[1].set_ylabel('Magnesium', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['Al'], ax = axes[0])

axes[0].set_xlabel('Aluminum', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.violinplot(x = 'Type', y = 'Al', data = dataset, hue = 'Type', dodge = False, ax = axes[1])

axes[1].set_xlabel('Type of glass', fontsize=14)

axes[1].set_ylabel('Aluminum', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['Si'], ax = axes[0])

axes[0].set_xlabel('Silicon', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.violinplot(x = 'Type', y = 'Si', data = dataset, hue = 'Type', dodge = False, ax = axes[1])

axes[1].set_xlabel('Type of glass', fontsize=14)

axes[1].set_ylabel('Silicon', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['K'], ax = axes[0])

axes[0].set_xlabel('Potassium', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.violinplot(x = 'Type', y = 'K', data = dataset, hue = 'Type', dodge = False, ax = axes[1])

axes[1].set_xlabel('Type of glass', fontsize=14)

axes[1].set_ylabel('Potassium', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['Ca'], ax = axes[0])

axes[0].set_xlabel('Calcium', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.violinplot(x = 'Type', y = 'Ca', data = dataset, hue = 'Type', dodge = False, ax = axes[1])

axes[1].set_xlabel('Type of glass', fontsize=14)

axes[1].set_ylabel('Calcium', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['Ca'], ax = axes[0])

axes[0].set_xlabel('Barium', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.violinplot(x = 'Type', y = 'Ca', data = dataset, hue = 'Type', dodge = False, ax = axes[1])

axes[1].set_xlabel('Type of glass', fontsize=14)

axes[1].set_ylabel('Barium', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['Fe'], ax = axes[0])

axes[0].set_xlabel('Iron', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.violinplot(x = 'Type', y = 'Fe', data = dataset, hue = 'Type', dodge = False, ax = axes[1])

axes[1].set_xlabel('Type of glass', fontsize=14)

axes[1].set_ylabel('Iron', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
X = dataset.drop('Type', axis = 1).values

y = dataset['Type'].values.reshape(-1,1)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
print("Shape of X_train: ",X_train.shape)

print("Shape of X_test: ", X_test.shape)

print("Shape of y_train: ",y_train.shape)

print("Shape of y_test",y_test.shape)
# Fitting Logistic Regression to the Training set

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression



classifier_lr = LogisticRegression()

steps = [

    ('scalar', StandardScaler()),

    ('model', LogisticRegression())

]



lr_pipe = Pipeline(steps)
parameters = { 'model__C' : [1,10,100,1000,10000],

               'model__fit_intercept' : [True],

               'model__multi_class' : ['auto'],

               'model__tol' : [0.0001],

               'model__solver' : ['newton-cg', 'lbfgs', 'sag', 'saga'],

               'model__n_jobs' : [-1],

               'model__max_iter' : [5000],

               'model__random_state': [42] 

}

classifier_lr = GridSearchCV(lr_pipe, parameters, iid=False, cv = 3)

classifier_lr = classifier_lr.fit(X_train, y_train.ravel())
from sklearn.metrics import accuracy_score



y_pred_lr_train = classifier_lr.predict(X_train)

accuracy_lr_train = accuracy_score(y_train, y_pred_lr_train)

print("Training set: ", accuracy_lr_train)



y_pred_lr_test = classifier_lr.predict(X_test)

accuracy_lr_test = accuracy_score(y_test, y_pred_lr_test)

print("Test set: ", accuracy_lr_test)
from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test, y_pred_lr_test), annot=True, cmap = 'viridis', fmt='.0f')

plt.show()
# Fitting classifier to the Training set

from sklearn.neighbors import KNeighborsClassifier

classifier_knn = KNeighborsClassifier()

steps = [

    ('scalar', StandardScaler()),

    ('model', KNeighborsClassifier())

]

knn_pipe = Pipeline(steps)
parameters = { 'model__algorithm' : ['brute'],

               'model__leaf_size' : [30,50,70,90,110],

               'model__metric' : ['minkowski'],

               'model__p' : [1],

               'model__n_neighbors' : [3,5,11,19],

               'model__weights' : ['uniform', 'distance'],

               'model__n_jobs' : [-1]

}

classifier_knn = GridSearchCV(knn_pipe, parameters, iid=False, cv = 3)

classifier_knn = classifier_knn.fit(X_train, y_train.ravel())
y_pred_knn_train = classifier_knn.predict(X_train)

accuracy_knn_train = accuracy_score(y_train, y_pred_knn_train)

print("Training set: ", accuracy_knn_train)



y_pred_knn_test = classifier_knn.predict(X_test)

accuracy_knn_test = accuracy_score(y_test, y_pred_knn_test)

print("Test set: ", accuracy_knn_test)
from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test, y_pred_knn_test), annot=True, cmap = 'viridis', fmt='.0f')

plt.show()
from sklearn.svm import SVC

classifier_svm = SVC()

steps = [

    ('scalar', StandardScaler()),

    ('model', SVC())

]

svm_linear_pipe = Pipeline(steps)
parameters = { 'model__kernel' : ['linear'],

               'model__C' : [1,10,100,1000,10000],

               'model__random_state' : [42]

}

classifier_svm_linear = GridSearchCV(svm_linear_pipe, parameters, iid=False, cv = 3)

classifier_svm_linear = classifier_svm_linear.fit(X_train, y_train.ravel())
y_pred_svm_linear_train = classifier_svm_linear.predict(X_train)

accuracy_svm_linear_train = accuracy_score(y_train, y_pred_svm_linear_train)

print("Training set: ", accuracy_svm_linear_train)



y_pred_svm_linear_test = classifier_svm_linear.predict(X_test)

accuracy_svm_linear_test = accuracy_score(y_test, y_pred_svm_linear_test)

print("Test set: ", accuracy_svm_linear_test)
from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test, y_pred_svm_linear_test), annot=True, cmap = 'viridis', fmt='.0f')

plt.show()
# Fitting classifier to the Training set

from sklearn.svm import SVC

classifier_svm_kernel = SVC()

steps = [

    ('scalar', StandardScaler()),

    ('model', SVC())

]

svm_kernel_pipe = Pipeline(steps)
parameters = { 'model__kernel' : ['rbf', 'poly', 'sigmoid'],

               'model__C' : [1,10,100,1000,10000],

               'model__gamma' : [0.001, 0.01, 0.1, 1, 'scale'],

               'model__random_state' : [42],

               'model__degree' : [1,2,3]

}

classifier_svm_kernel = GridSearchCV(svm_kernel_pipe, parameters, iid=False, cv = 3)

classifier_svm_kernel = classifier_svm_kernel.fit(X_train, y_train.ravel())
y_pred_svm_kernel_train = classifier_svm_kernel.predict(X_train)

accuracy_svm_kernel_train = accuracy_score(y_train, y_pred_svm_kernel_train)

print("Training set: ", accuracy_svm_kernel_train)



y_pred_svm_kernel_test = classifier_svm_kernel.predict(X_test)

accuracy_svm_kernel_test = accuracy_score(y_test, y_pred_svm_kernel_test)

print("Test set: ", accuracy_svm_kernel_test)
from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test, y_pred_svm_kernel_test), annot=True, cmap = 'viridis', fmt='.0f')

plt.show()
# Fitting classifier to the Training set

from sklearn.naive_bayes import GaussianNB

classifier_nb = GaussianNB()

classifier_nb.fit(X_train, y_train.ravel())
y_pred_nb_train = classifier_nb.predict(X_train)

accuracy_nb_train = accuracy_score(y_train, y_pred_nb_train)

print("Training set: ", accuracy_nb_train)



y_pred_nb_test = classifier_nb.predict(X_test)

accuracy_nb_test = accuracy_score(y_test, y_pred_nb_test)

print("Test set: ", accuracy_nb_test)
from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test, y_pred_nb_test), annot=True, cmap = 'viridis', fmt='.0f')

plt.show()
# Fitting classifier to the Training set

from sklearn.tree import DecisionTreeClassifier

classifier_dt = DecisionTreeClassifier()



steps = [

    ('scalar', StandardScaler()),

    ('model', DecisionTreeClassifier())

]

dt_pipe = Pipeline(steps)
# Applying Grid Search to find the best model and the best parameters

parameters = [ { "model__max_depth": np.arange(1,21),

                 "model__min_samples_leaf": [1, 5, 10, 20, 50, 100],

                 "model__min_samples_split": np.arange(2, 11),

                 "model__criterion": ["gini"],

                 "model__random_state" : [42]}

            ]

classifier_dt = GridSearchCV(estimator = dt_pipe,

                           param_grid  = parameters,

                           cv = 3,

                           iid = False,

                           n_jobs = -1)

classifier_dt = classifier_dt.fit(X_train, y_train.ravel())
y_pred_dt_train = classifier_dt.predict(X_train)

accuracy_dt_train = accuracy_score(y_train, y_pred_dt_train)

print("Training set: ", accuracy_dt_train)



y_pred_dt_test = classifier_dt.predict(X_test)

accuracy_dt_test = accuracy_score(y_test, y_pred_dt_test)

print("Test set: ", accuracy_dt_test)
from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test, y_pred_dt_test), annot=True, cmap = 'viridis', fmt='.0f')

plt.show()
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

classifier_rf = RandomForestClassifier()



steps = [

    ('scalar', StandardScaler()),

    ('model', RandomForestClassifier())

]

rf_pipe = Pipeline(steps)
parameters =  { "model__n_estimators": [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],

                "model__max_features": ["auto", "sqrt"],

                "model__max_depth": np.linspace(10, 110, num = 11),

                "model__min_samples_split": [2, 5, 10],

                "model__min_samples_leaf": [1, 2, 4],

                "model__bootstrap": [True, False],

                "model__criterion": ["gini"],

                "model__random_state" : [42] }

            

classifier_rf = RandomizedSearchCV(estimator = rf_pipe,

                                  param_distributions = parameters,

                                  n_iter = 100,

                                  cv = 3,

                                  random_state=42,

                                  verbose = 4,

                                  n_jobs = -1)

classifier_rf = classifier_rf.fit(X_train, y_train.ravel())
y_pred_rf_train = classifier_rf.predict(X_train)

accuracy_rf_train = accuracy_score(y_train, y_pred_rf_train)

print("Training set: ", accuracy_rf_train)



y_pred_rf_test = classifier_rf.predict(X_test)

accuracy_rf_test = accuracy_score(y_test, y_pred_rf_test)

print("Test set: ", accuracy_rf_test)
from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test, y_pred_rf_test), annot=True, cmap = 'viridis', fmt='.0f')

plt.show()
# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense
# Feature Scaling

sc_X = StandardScaler()

X_train_scaled = sc_X.fit_transform(X_train)

X_test_scaled = sc_X.fit_transform(X_test)

print(X_train_scaled.shape)

print(X_test_scaled.shape)
# Defining a function to encode output column

from keras.utils import to_categorical

def encode(data):

    print('Shape of data (BEFORE encode): %s' % str(data.shape))

    encoded = to_categorical(data)

    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))

    return encoded
y_train_encoded = encode(y_train)
y_test_encoded = encode(y_test)
y_train_encoded = np.delete(y_train_encoded, [0,4], axis = 1)

y_test_encoded = np.delete(y_test_encoded, [0,4], axis = 1)

print(y_train_encoded[2])

print(y_test_encoded[2])
# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the second hidden layer

classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

history = classifier.fit(X_train_scaled, y_train_encoded, validation_data=(X_test_scaled, y_test_encoded), batch_size = 100, epochs = 1150)
f, axes = plt.subplots(1,2,figsize=(14,4))



axes[0].plot(history.history['loss'])

axes[0].plot(history.history['val_loss'])

axes[0].set_xlabel('Loss', fontsize=14)

axes[0].set_ylabel('Epuch', fontsize=14)

axes[0].yaxis.tick_left()

axes[0].legend(['Train', 'Test'], loc='upper left')



axes[1].plot(history.history['acc'])

axes[1].plot(history.history['val_acc'])

axes[1].set_xlabel('Accuracy', fontsize=14)

axes[1].set_ylabel('Epoch', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(['Train', 'Test'], loc='upper left')



plt.show()
print("Training set: ", history.history.get('acc')[-1])

print("Test set: ", history.history.get('val_acc')[-1])
models = [('Logistic Regression', accuracy_lr_train, accuracy_lr_test),

          ('KNN', accuracy_knn_train, accuracy_knn_test),

          ('SVM (Linear)', accuracy_svm_linear_train, accuracy_svm_linear_test),

          ('SVM (Kernel)', accuracy_svm_kernel_train, accuracy_svm_kernel_test),

          ('Naive Bayes', accuracy_nb_train, accuracy_nb_test),

          ('Decision Tree Classification', accuracy_dt_train, accuracy_dt_test),

          ('Random Forest Classification', accuracy_rf_train, accuracy_rf_test),

          ('ANN', history.history.get('acc')[-1], history.history.get('val_acc')[-1]),

         ]
predict = pd.DataFrame(data = models, columns=['Model', 'Training Accuracy', 'Test Accuracy'])

predict
f, axes = plt.subplots(2,1, figsize=(14,10))



predict.sort_values(by=['Training Accuracy'], ascending=False, inplace=True)



sns.barplot(x='Training Accuracy', y='Model', data = predict, palette='Blues_d', ax = axes[0])

#axes[0].set(xlabel='Region', ylabel='Charges')

axes[0].set_xlabel('Training Accuracy', size=16)

axes[0].set_ylabel('Model')

axes[0].set_xlim(0,1.0)

axes[0].set_xticks(np.arange(0, 1.1, 0.1))



predict.sort_values(by=['Test Accuracy'], ascending=False, inplace=True)



sns.barplot(x='Test Accuracy', y='Model', data = predict, palette='Greens_d', ax = axes[1])

#axes[0].set(xlabel='Region', ylabel='Charges')

axes[1].set_xlabel('Test Accuracy', size=16)

axes[1].set_ylabel('Model')

axes[1].set_xlim(0,1.0)

axes[1].set_xticks(np.arange(0, 1.1, 0.1))



plt.show()