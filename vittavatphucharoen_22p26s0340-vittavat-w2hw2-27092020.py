import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Import Dataset

ttn_trn = pd.read_csv('../input/titanic/train.csv')

ttn_tst = pd.read_csv('../input/titanic/test.csv')

## Explore Training Set

ttn_trn.info()
pd.concat([ttn_trn.head(), ttn_trn.tail()])
ttn_trn.describe()
ttn_trn.Survived.value_counts()
ttn_trn[ttn_trn['Age'].isna()]
ttn_trn[ttn_trn['Age'].isna()][ttn_trn[ttn_trn['Age'].isna()]['Sex']=='female'].head()
## Survivor in each Pclass

ttn_trn.groupby(by=['Pclass','Survived'])['Pclass'].count().unstack('Survived').plot(kind='bar',stacked=True)
ttn_trn.groupby(by=['Sex','Survived'])['Sex'].count().unstack('Survived').plot(kind='bar',stacked=True)
ttn_trn.groupby(by=['SibSp','Survived'])['SibSp'].count().unstack('Survived').plot(kind='bar',stacked=True)
ttn_trn.groupby(by=['Embarked','Survived'])['Embarked'].count().unstack('Survived').plot(kind='bar',stacked=True)
ttn_trn.groupby(by=['Embarked']).mean().drop('PassengerId', axis=1)
ttn_trn.groupby(by=['Survived','SibSp'])['Survived'].count().unstack('SibSp').transpose().plot(kind='line')
ttn_trn[ttn_trn['SibSp']!=0]
ttn_trn.groupby(by=['Survived','Parch'])['Survived'].count().unstack('Parch').transpose().plot(kind='line')
ttn_trn[ttn_trn['Parch']!=0]
## Drop Cabin Columns and Missing Value in Age and Embarked Columns.

ttn_trn_cleaned = ttn_trn.drop(['Cabin','Ticket','Name','PassengerId'], axis=1).dropna().reset_index(drop=True)
ttn_trn_cleaned.head()
## Create One hot Encode Columns

cate_vars = ['Pclass','Sex','Embarked']

ttn_trn_cln_dum = pd.get_dummies(ttn_trn_cleaned, columns = cate_vars)

ttn_trn_cln_dum.head()
from sklearn.model_selection import KFold
## K-folds Cross validation

X = ttn_trn_cln_dum.drop('Survived', axis = 1)

y = ttn_trn_cln_dum['Survived']

kf = KFold(n_splits = 5)
X.info()
from sklearn.metrics import recall_score, precision_score, accuracy_score

from sklearn import preprocessing
## Decision Tree

from sklearn import tree
## Decision Tree Model

n = 0

folds = []

model_dct_ls = []

recall_dct_ls = []

precision_dct_ls = []

F_measure_dct_ls = []



for train_index, test_index in kf.split(X):

    n = n + 1

    

    # Identify train and test set

    X_train, X_test = X.loc[train_index,], X.loc[test_index,]

    y_train, y_test = y.loc[train_index,], y.loc[test_index,]

    print("Folds:",n,"/",kf.get_n_splits(X))

    

    # Build Model and Predict

    clasfr = tree.DecisionTreeClassifier() ## Please Check Classifier before use this loop!!

    model = clasfr.fit(X_train, y_train)

    y_hat = model.predict(X_test)

    

    # Calculate Model Measure

    recall = recall_score(y_test, y_hat, average = 'binary')

    precision = precision_score(y_test, y_hat, average = 'binary')

    F_measure = 2 * (precision * recall) / (precision + recall)

    

    # Keep Loop Product

    folds.append(n)

    model_dct_ls.append(model)

    recall_dct_ls.append(recall)

    precision_dct_ls.append(precision)

    F_measure_dct_ls.append(F_measure)

    

    if n == kf.get_n_splits(X):

        print("Finish!!")
## Create Dataframe for show

metric_dct = pd.DataFrame({'Fold': folds,

                          'Recall': recall_dct_ls,

                          'Precision': precision_dct_ls,

                          'F_Measure': F_measure_dct_ls}).set_index(['Fold'])



av_F1_dct = metric_dct['F_Measure'].mean()



## Show Metrics

print("Classifier: Decision Tree")

print("")

print(metric_dct)

print("")

print("Average F-Measure:",av_F1_dct)
## Naive Bayes

from sklearn.naive_bayes import GaussianNB
## Normalize X

scaler = preprocessing.MinMaxScaler()

X = scaler.fit_transform(ttn_trn_cln_dum.drop('Survived', axis = 1))

X = pd.DataFrame(X)
## Train Naive Bayes Model

n = 0

folds = []

model_nb_ls = []

recall_nb_ls = []

precision_nb_ls = []

F_measure_nb_ls = []



for train_index, test_index in kf.split(X):

    n = n + 1

    

    # Identify train and test set

    X_train, X_test = X.loc[train_index,], X.loc[test_index,]

    y_train, y_test = y.loc[train_index,], y.loc[test_index,]

    print("Folds:",n,"/",kf.get_n_splits(X))

    

    # Build Model and Predict

    clasfr = GaussianNB() ## Please Check Classifier before use this loop!!

    model = clasfr.fit(X_train, y_train)

    y_hat = model.predict(X_test)

    

    # Calculate Model Measure

    recall = recall_score(y_test, y_hat, average = 'binary')

    precision = precision_score(y_test, y_hat, average = 'binary')

    F_measure = 2 * (precision * recall) / (precision + recall)

    

    # Keep Loop Product

    folds.append(n)

    model_nb_ls.append(model)

    recall_nb_ls.append(recall)

    precision_nb_ls.append(precision)

    F_measure_nb_ls.append(F_measure)

    

    if n == kf.get_n_splits(X):

        print("Finish!!")
## Create Dataframe for show

metric_nb = pd.DataFrame({'Fold': folds,

                          'Recall': recall_nb_ls,

                          'Precision': precision_nb_ls,

                          'F_Measure': F_measure_nb_ls}).set_index(['Fold'])



av_F1_nb = metric_nb['F_Measure'].mean()



## Show Metrics

print("Classifier: Naive Bayes")

print("")

print(metric_nb)

print("")

print("Average F-Measure:",av_F1_nb)
## Neural Network

import tensorflow as tf
X = scaler.fit_transform(ttn_trn_cln_dum.drop('Survived', axis = 1))

X = pd.DataFrame(X)
n = 0

folds = []

model_nn_ls = []

acc_nn_ls = []

recall_nn_ls = []

precision_nn_ls = []

F_measure_nn_ls = []





dimension = (X.shape[1],)



model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(12, input_shape = dimension))

model.add(tf.keras.layers.Dense(120, activation = 'relu'))

model.add(tf.keras.layers.Dense(240, activation = 'relu'))

model.add(tf.keras.layers.Dense(360, activation = 'relu'))

model.add(tf.keras.layers.Dense(360, activation = 'relu'))

model.add(tf.keras.layers.Dense(240, activation = 'relu'))

model.add(tf.keras.layers.Dense(48, activation = 'relu'))

model.add(tf.keras.layers.Dense(12, activation = 'relu'))

model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))



model.compile(loss = tf.keras.losses.MeanSquaredError(),

              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),

              metrics = ['accuracy'])



#model.compile(loss = tf.keras.losses.BinaryCrossentropy(),

#              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))



for train_index, test_index in kf.split(X):

    n = n + 1

    

    # Identify train and test set

    X_train, X_test = X.loc[train_index,], X.loc[test_index,]

    y_train, y_test = y.loc[train_index,], y.loc[test_index,]

    print("Folds:",n,"/",kf.get_n_splits(X))

    

    # Fit Model and Predict

    model.fit(X_train, y_train, epochs = 200)

    y_hat = model.predict(X_test)

    y_hat = y_hat.round()



    # Calculate Model Measure

    acc = accuracy_score(y_test, y_hat)

    recall = recall_score(y_test, y_hat, average = 'binary')

    precision = precision_score(y_test, y_hat, average = 'binary')

    F_measure = 2 * (precision * recall) / (precision + recall)

    

    # Keep Loop Product

    folds.append(n)

    model_nn_ls.append(model)

    acc_nn_ls.append(acc)

    recall_nn_ls.append(recall)

    precision_nn_ls.append(precision)

    F_measure_nn_ls.append(F_measure)

    

    if n == kf.get_n_splits(X):

        print("Finish!!")

print("Finish!!")
## Create Dataframe for show

metric_nn = pd.DataFrame({'Fold': folds,

                          'Recall': recall_nn_ls,

                          'Precision': precision_nn_ls,

                          'F_Measure': F_measure_nn_ls}).set_index(['Fold'])



av_F1_nn = metric_nn['F_Measure'].mean()



## Show Metrics

print("Classifier: Neural Network")

print("")

print(metric_nn)

print("")

print("Average F-Measure:",av_F1_nn)

{'Accuracy': acc_nn_ls}
## Recall

## Precision

## F-Measure

## Average F-Measure ของทั้งชุดข้อมูล

av_F1_all = pd.DataFrame({'Classifier': ['Decision Tree','Naive Bayes','Neural Network'],

              'Average F-Measure': [av_F1_dct,av_F1_nb,av_F1_nn]}).set_index('Classifier')
## Build Dataframe to Compare Model Measurement and Summary

summary = pd.concat({'Decision Tree':metric_dct,'Naive Bayes':metric_nb,'Neural Network':metric_nn})

print(summary)

print('')

print('------------------------------------------------------------------')

print('')

print(av_F1_all)