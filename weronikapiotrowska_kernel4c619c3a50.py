# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import keras

from keras import backend as K



from keras import Sequential

from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, precision_score

from sklearn.metrics import recall_score, f1_score, accuracy_score

from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/beers.csv')

train_df = train_df.drop(['Name'], axis=1)

print(train_df.columns.values)

train_df.head()
train_df.info()
train_df = train_df.drop(['PitchRate', 'UserId'], axis=1)

train_df.describe()
train_df.describe(include=['O'])
sns.heatmap(train_df.corr(), annot=True)

plt.tight_layout()
sns.boxplot(x='Style', y='Color', data=train_df);
sns.boxplot(x='Style', y='BoilTime', data=train_df);
age_mean = np.mean(train_df['BoilGravity'])

train_df['BoilGravity'] = train_df[['BoilGravity']].fillna(age_mean)
categoricals = list(train_df.select_dtypes(include=['O']).columns)

encoder = OneHotEncoder(sparse=False)

encoded = encoder.fit_transform(train_df[categoricals])

train_ohe = pd.DataFrame(encoded, columns=np.hstack(encoder.categories_))

train_df = pd.concat((train_df, train_ohe), axis=1).drop(categoricals, axis=1)



Y = train_ohe.values



X = train_df.values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

train_df.head()



print(train_df.info())

model = Sequential()

model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(32, activation='relu', kernel_initializer='random_normal'))

model.add(Dropout(0.5))

model.add(Dense(16, activation='sigmoid', kernel_initializer='random_normal'))

model.add(Dropout(0.5))

model.add(Dense(8, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(4, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(7, activation='sigmoid'))



model.summary()

model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

train_df.head()
history = model.fit(X_train, y_train, batch_size=256, epochs=100)



sns.lineplot(range(len(history.history['loss'])), history.history['loss'])





        
sns.lineplot(range(len(history.history['accuracy'])), history.history['accuracy'])

pred = model.predict(X_test) > 0.5

confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))

acc = accuracy_score(y_test, pred)





print('Accuracy: {}'.format(

        acc))