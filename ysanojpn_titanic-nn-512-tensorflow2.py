# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pathlib

import matplotlib.pyplot as plt

import seaborn as sns
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



print(tf.__version__)
# Use some functions from tensorflow_docs

!pip install -q git+https://github.com/tensorflow/docs

import tensorflow_docs as tfdocs

import tensorflow_docs.plots

import tensorflow_docs.modeling
#import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

raw_test = pd.read_csv("../input/titanic/test.csv")

raw_dataset = pd.read_csv("../input/titanic/train.csv")



dataset = raw_dataset.copy()

quizset = raw_test.copy()

provset = dataset.append(pd.concat([quizset,gender_submission['Survived']],axis=1,sort=False)).reset_index(drop=True)

datacount = len(dataset)



print('dataset:', len(dataset))  # DataFrame of training and validation

print('quizset:', len(quizset))  # DataFrame of quiz(Survived columns not exist)

print('provset: ',len(provset))  # DataFrame of provisioning (Survived columns NOT valid)
provset.dtypes
provset.isna().sum()
# Many data is missing in Age but Name's title(Mr,Mrs,Master,etc..) will fill this gap a little.

provset['Title'] = provset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



# Titles seem to determine their fate

pd.crosstab(provset['Title'],provset['Sex'])
# Age mean,median per Title

age_mean = provset.groupby(['Title'])['Age'].mean()

age_median = provset.groupby(['Title'])['Age'].median()

age_std = provset.groupby(['Title'])['Age'].std()

pd.DataFrame([age_mean,age_median,age_std],index=['mean','median','std'])

# Missing Data fill with median per Title

for title,subset in provset.groupby(['Title']):

    provset.loc[provset['Age'].isna() & (provset['Title'] == title), 'Age'] = age_median[title]

provset            
# Cabin to Deck

provset['Deck'] = provset['Cabin'].str.extract('([A-Z])')



# Deck and Survived

pd.crosstab(provset['Deck'],provset['Pclass'])
# fill na class

provset.loc[provset['Embarked'].isna(), 'Embarked'] = 'O' # other class

#provset.loc[provset['Cabin'].isna(), 'Cabin'] = 'O' # other class

provset.loc[provset['Deck'].isna(), 'Deck'] = 'O' # other class
# add family_size

provset['FamilySize'] = provset['Parch'] + provset['SibSp'] + 1



# and add IsAlone

provset['IsAlone'] = 0

provset.loc[(provset['FamilySize'] == 1), 'IsAlone'] = 1



# https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut

# add FareBin (qcut)

provset['FareBin'] = pd.qcut(provset['Fare'], 4)

# AgeBin (cut)

provset['AgeBin'] = pd.cut(provset['Age'].astype(int), 5)



provset
# drop unreasonable columns

#provset = provset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

provset.tail()
dataset = provset[:datacount]

temp = dataset.copy()

dataset_labels = temp.pop('Survived')

sns.pairplot(dataset[["Survived", "Age", "Fare", "FamilySize"]], diag_kind="kde")
#graph distribution of quantitative data

plt.figure(figsize=[16,12])



plt.subplot(231)

plt.boxplot(x=dataset['Fare'], showmeans = True, meanline = True)

plt.title('Fare Boxplot')

plt.ylabel('Fare ($)')



plt.subplot(232)

plt.boxplot(dataset['Age'], showmeans = True, meanline = True)

plt.title('Age Boxplot')

plt.ylabel('Age (Years)')



plt.subplot(233)

plt.boxplot(dataset['FamilySize'], showmeans = True, meanline = True)

plt.title('Family Size Boxplot')

plt.ylabel('Family Size (#)')



plt.subplot(234)

plt.hist(x = [dataset[dataset['Survived']==1]['Fare'], dataset[dataset['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(235)

plt.hist(x = [dataset[dataset['Survived']==1]['Age'], dataset[dataset['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(236)

plt.hist(x = [dataset[dataset['Survived']==1]['FamilySize'], dataset[dataset['Survived']==0]['FamilySize']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Family Size Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')

plt.legend()
#graph individual features by survival

fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.barplot(x = 'Embarked', y = 'Survived', order=['S','C','Q'], data=dataset, ax = saxis[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=dataset, ax = saxis[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=dataset, ax = saxis[0,2])



sns.pointplot(x = 'FareBin', y = 'Survived',  data=dataset, ax = saxis[1,0])

sns.pointplot(x = 'AgeBin', y = 'Survived',  data=dataset, ax = saxis[1,1])

sns.pointplot(x = 'FamilySize', y = 'Survived', data=dataset, ax = saxis[1,2])
# one hot encoding

provset = pd.get_dummies(provset, columns=['Sex','Embarked','Pclass','Title', 'Deck', 'FareBin', 'AgeBin'])

# dataset

dataset = provset[:datacount]

dataset_labels = dataset.pop('Survived')



# drop unreasonable columns

dataset_x = dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)



# quizset needs PassengerId

quizset = provset[datacount:].drop('Survived',axis=1)

quizset_x = quizset.drop(['Name', 'Ticket', 'Cabin'], axis=1)
NUMERICAL_COLUMNS = ['Age', 'Fare', 'Parch', 'SibSp', 'FamilySize']

provset_stats = provset[NUMERICAL_COLUMNS].describe().T

provset_stats
dataset_x.loc[:,'Age']

#provset_stats.loc['Age','mean']

#provset_stats.loc['Age','std']
def norm(dataset_x):

    for col in NUMERICAL_COLUMNS:

        x = dataset_x.loc[:,col]

        mean = provset_stats.loc[col,'mean']

        std = provset_stats.loc[col,'std']

        dataset_x.loc[:,col] = (x - mean) / std

    return dataset_x

    

normed_dataset_x = norm(dataset_x)

normed_quizset_x = norm(quizset_x)

normed_dataset_x
def build_model(units=512, activation="relu", optimizer="adam", l2_param=0.001, dropout_rate=0.2, learning_rate=1e-05):

    model = keras.Sequential([

        layers.Dense(units,

                     kernel_regularizer=tf.keras.regularizers.l2(l2_param),

                     activation=activation, input_shape=[len(dataset_x.keys())]),

        layers.Dropout(dropout_rate),

        layers.Dense(units,

                     kernel_regularizer=tf.keras.regularizers.l2(l2_param),

                     activation=activation),

        layers.Dropout(dropout_rate),

        layers.Dense(units,

                     kernel_regularizer=tf.keras.regularizers.l2(l2_param),

                     activation=activation),

        layers.Dropout(dropout_rate),

        layers.Dense(1, activation='sigmoid')

    ])



    optimizer = tf.keras.optimizers.get(optimizer)

    optimizer.learning_rate=learning_rate



    model.compile(loss='binary_crossentropy',

        optimizer=optimizer,

        metrics=['accuracy', 'binary_crossentropy'])

    return model



model = build_model()
model.summary()
ex_batch = normed_dataset_x[:10]

ex_result = model.predict(ex_batch)

ex_result
EPOCHS = 600



history = model.fit(

  normed_dataset_x, dataset_labels,

  epochs=EPOCHS, validation_split = 0.25, verbose=0,

  callbacks=[tfdocs.modeling.EpochDots()])
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric = "accuracy")

plt.ylim([0.7, 1])

plt.ylabel('Accuracy')
plotter.plot({'Basic': history}, metric = "binary_crossentropy")

plt.ylim([0.15, 0.6])

plt.ylabel('Binary Crossentropy')
model = build_model()



# The patience parameter is the amount of epochs to check for improvement

early_stop = keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=60, restore_best_weights=True)



early_history = model.fit(normed_dataset_x, dataset_labels, 

                    epochs=EPOCHS, validation_split = 0.25, verbose=0, 

                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])
plotter.plot({'Early Stopping': early_history}, metric = "binary_crossentropy")

plt.ylim([0.2, 0.7])

plt.ylabel('Binary Crossentropy')
dataset_predictions = model.predict(normed_dataset_x).flatten()

error = dataset_predictions - dataset_labels



_ = plt.hist(abs(error),bins=100)

#_ = plt.scatter(dataset_predictions,dataset_labels, s=100, alpha=0.2)
# quiz_data

quizset_x_copy = quizset_x.copy()

pid = quizset_x_copy.pop('PassengerId')



yhat = model.predict(quizset_x_copy)



submission = np.concatenate([[pid, yhat.flatten()]])

submission = pd.DataFrame(submission.T, columns=['PassengerId','Survived'])

submission.loc[:,'Survived'] = submission.loc[:,'Survived'].apply(lambda x: 1 if x > 0.5 else 0)

submission = submission.astype(np.int,copy=False)

submission
!rm ./submission.csv

submission.to_csv('submission.csv',index=False)