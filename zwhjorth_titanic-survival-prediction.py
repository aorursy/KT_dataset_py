import pandas as pd

import numpy as np

from sklearn import preprocessing

from keras.models import Sequential

from keras.layers import Dense



df = pd.read_csv('../input/titanic/train.csv')

df.drop(['Ticket','Name'],axis=1,inplace=True)

df['Cabin'] = df['Cabin'].fillna(0)

new_Cabin = []



for cabin in df['Cabin']:

    if type(cabin) != int:

        new = cabin[0]

        if new == 'A':

            new = 1

        if new == 'B':

            new = 2

        if new == 'C':

            new = 3

        if new == 'D':

            new = 4

        if new == 'E':

            new = 5

        if new == 'F':

            new = 6

        if new == 'G':

            new = 7

        if new == 'T':

            new = 8

    else:

        new = cabin

    new_Cabin.append(new)

df.loc[:,'Cabin'] = new_Cabin

df['Cabin'] = df['Cabin'].replace('0',np.nan)

df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mean())

df['Sex'] = df['Sex'].replace('male',np.nan)

df['Sex'] = df['Sex'].fillna(0)

df['Sex'] = df['Sex'].replace('female',np.nan)

df['Sex'] = df['Sex'].fillna(1)

df['Embarked'] = df['Embarked'].replace('Q',np.nan)

df['Embarked'] = df['Embarked'].fillna(0)

df['Embarked'] = df['Embarked'].replace('S',np.nan)

df['Embarked'] = df['Embarked'].fillna(1)

df['Embarked'] = df['Embarked'].replace('C',np.nan)

df['Embarked'] = df['Embarked'].fillna(2)

df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Fare'] = df['Fare'].fillna(df['Fare'].mean())



df_scaled = preprocessing.scale(df)

df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

df_scaled['Survived'] = df['Survived']

df = df_scaled



X = df.loc[:,df.columns != 'Survived']

y = df.loc[:, 'Survived']



model = Sequential()

model.add(Dense(32,activation = 'relu', input_dim = 9))

model.add(Dense(16, activation = 'relu'))

model.add(Dense(8, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



model.fit(X, y, epochs=250)



df2 = pd.read_csv('../input/titanic/test.csv')

df2.drop(['Ticket','Name'],axis=1,inplace=True)

df2['Cabin'] = df2['Cabin'].fillna(0)

new_Cabin = []



for cabin in df2['Cabin']:

    if type(cabin) != int:

        new = cabin[0]

        if new == 'A':

            new = 1

        if new == 'B':

            new = 2

        if new == 'C':

            new = 3

        if new == 'D':

            new = 4

        if new == 'E':

            new = 5

        if new == 'F':

            new = 6

        if new == 'G':

            new = 7

        if new == 'T':

            new = 8

    else:

        new = cabin

    new_Cabin.append(new)

df2.loc[:,'Cabin'] = new_Cabin

df2['Cabin'] = df2['Cabin'].replace('0',np.nan)

df2['Cabin'] = df2['Cabin'].fillna(df2['Cabin'].mean())

df2['Sex'] = df2['Sex'].replace('male',np.nan)

df2['Sex'] = df2['Sex'].fillna(0)

df2['Sex'] = df2['Sex'].replace('female',np.nan)

df2['Sex'] = df2['Sex'].fillna(1)

df2['Embarked'] = df2['Embarked'].replace('Q',np.nan)

df2['Embarked'] = df2['Embarked'].fillna(0)

df2['Embarked'] = df2['Embarked'].replace('S',np.nan)

df2['Embarked'] = df2['Embarked'].fillna(1)

df2['Embarked'] = df2['Embarked'].replace('C',np.nan)

df2['Embarked'] = df2['Embarked'].fillna(2)

df2['Age'] = df2['Age'].fillna(df2['Age'].mean())

df2['Fare'] = df2['Fare'].fillna(df2['Fare'].mean())



df2_scaled = preprocessing.scale(df2)

df2_scaled = pd.DataFrame(df2_scaled, columns=df2.columns)

df3 = df2

df2 = df2_scaled



prediction = model.predict(df2)

prediction = pd.DataFrame(prediction, columns = ['Survived'])

prediction.loc[prediction['Survived'] > 0.5] = 1

prediction.loc[prediction['Survived'] <= 0.5] = 0

prediction.Survived.astype(int)

pd.DataFrame({'PassengerId':df3.PassengerId, 'Survived':prediction.Survived}).to_csv('titanic_submission_zwhjorth.csv',index=False)