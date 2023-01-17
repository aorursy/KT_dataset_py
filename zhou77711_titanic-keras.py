import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

print(os.listdir("../input"))

from pathlib import Path
import keras

from keras.models import Sequential

from keras.models import Model

from keras.layers import Dense, Input

from keras.utils import to_categorical

from tensorboardX import FileWriter, summary
path = Path('../input')

train_df = pd.read_csv(path/'train.csv')

print(train_df.head())
test_df = pd.read_csv(path/'test.csv')

test_df.head()
train_df.Embarked = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

train_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace=True)

train_df['Sex'] = train_df['Sex'].apply(lambda x: 1 if x=='male' else 0)

train_df.Embarked = train_df.Embarked.map({'S':0, 'C':1, 'Q':2}).astype(int)
test_df.Embarked = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

test_df['Sex'] = test_df['Sex'].apply(lambda x: 1 if x=='male' else 0)

test_df.Embarked = test_df.Embarked.map({'S':0, 'C':1, 'Q':2}).astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_pid = test_df.PassengerId
df = pd.concat([train_df, test_df], ignore_index=True)

print(len(df))

print(df.head())
guess_ages = np.zeros((2,3)) #Store the guess for each  Pclass and Sex

for dataset in [train_df, test_df]:

    for i in range(0, 2): # Sex has two values: 0 and 1

        for j in range(0, 3): # Pclass range from 1 to 3

            guess_df = df[(df['Sex'] == i) & (df['Pclass'] == j+1)]['Age'].dropna()

    #         print(i,j,guess_df)

            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5



    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)
for dataset in [train_df, test_df]:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
for dataset in [train_df, test_df]:

    dataset.loc[ dataset['Fare'] <= 128, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 128) & (dataset['Fare'] <= 256), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 256) & (dataset['Fare'] <= 384), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 384, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df.head()
train_data = train_df.drop(['Name', 'Ticket', 'Cabin','PassengerId'], axis=1)

train_data.head()
test_passengerID = test_df.PassengerId

test_data = test_df.drop(['Name', 'Ticket', 'Cabin','PassengerId'], axis = 1)

test_data.head()
print(len(train_data), len(test_data))
def split_by_rand_pct(idx_list:list, valid_pct:float=0.2, seed:int=None)->'List':

    "Split the items randomly by putting `valid_pct` in the validation set, optional `seed` can be passed."

    if valid_pct==0.: return

    if seed is not None: np.random.seed(seed)

    rand_idx = np.random.permutation(idx_list)

    cut = int(valid_pct * len(idx_list))

    return (rand_idx[cut:].tolist(), rand_idx[:cut].tolist())
train_data_idx, valid_data_idx = split_by_rand_pct(list(train_data.index),seed=2019)
train_1 = train_data.loc[train_data_idx,:]

train_label_1 = np.array(train_1.Survived)

train_1 = train_1.drop(['Survived'],axis=1)

train_array_1 = train_1.values

print(train_1.head())
valid = train_data.loc[valid_data_idx,:]

valid_label = np.array(valid.Survived)

valid = valid.drop(['Survived'],axis=1)

valid_array = valid.values

valid.head()
classifier_1 = Sequential()

classifier_1.add(Dense(output_dim = 4, init = "uniform", activation = 'relu', input_dim = 7))

classifier_1.add(Dense(output_dim = 2, init = "uniform", activation = 'relu'))

classifier_1.add(Dense(output_dim = 1, init = "uniform", activation = 'sigmoid'))

classifier_1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier_1.fit(train_array_1, train_label_1, batch_size = 10, nb_epoch = 100)
from sklearn.metrics import roc_auc_score

from sklearn import metrics

valid_pred = classifier_1.predict(valid_array)

auc = roc_auc_score(valid_label, valid_pred)

fpr, tpr, thresholds = metrics.roc_curve(valid_label, valid_pred, pos_label=1)
from matplotlib import pyplot

pyplot.plot([0, 1], [0, 1], linestyle='--')

pyplot.plot(fpr, tpr, marker='.')

pyplot.show()
import math

dist = math.sqrt(fpr[0]**2 + (1-tpr[0])**2)

ind  = 0

for i in range(1, len(fpr)):

    tmp = math.sqrt(fpr[i]**2+(1-tpr[i])**2)

    if dist > tmp:

        dist = tmp

        ind = i

thres = thresholds[ind]

thres
test = test_data.values

test[0,:].shape
pred_ann = classifier_1.predict(test)

len(pred_ann)
pred = [0]*len(pred_ann)

for i in range(len(pred_ann)):

    if pred_ann[i][0] >= thres:

        pred[i] = 1
submission_df = pd.DataFrame({'PassengerId': test_pid, 'Survived': pred}, columns=['PassengerId', 'Survived'])

submission_df.head()
submission_df.to_csv('TitanicSubmission.csv', index=False)