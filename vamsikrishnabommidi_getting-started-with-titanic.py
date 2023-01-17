# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head(10)
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()
train_data=train_data.drop(['PassengerId', 'Ticket','Cabin'],axis=1)

test_data=test_data.drop(['PassengerId', 'Ticket','Cabin'],axis=1)


test_data['Fare'].fillna(7,inplace=True)

test_data.isnull().sum()
train_data['Embarked'].fillna('S',inplace=True)
train_data['Sex']=train_data['Sex'].map({'male':1,'female':0})

test_data['Sex']=test_data['Sex'].map({'male':1,'female':0})
def title(x):

    if 'Mr.' in x.split():

        return 'Mr'

    elif 'Master.' in x.split():

        return 'Master'

    elif 'Miss.'  in x.split():

        return 'Miss'

    elif 'Mrs.' in x.split():

        return 'Mrs'

    else:

        return 'X'
train_data['Title']=train_data['Name'].apply(lambda x:title(x))

test_data['Title']=test_data['Name'].apply(lambda x:title(x))
train_data['Title'].value_counts()
train_data=train_data.drop('Name',axis=1)

test_data=test_data.drop('Name',axis=1)
train_data.groupby('Title').mean()['Age']
train_data['age'] = train_data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

test_data['age'] = test_data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
train_data=train_data.drop('Age',axis=1)

test_data=test_data.drop('Age',axis=1)
train_data=pd.get_dummies(train_data,drop_first=True)

test_data=pd.get_dummies(test_data,drop_first=True)
X=train_data.drop('Survived',axis=1).values

y=train_data['Survived'].values

test_data=test_data.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=85)
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)

test_data=scaler.transform(test_data)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.callbacks import EarlyStopping
#Estopper=EarlyStopping(patience=25,verbose=1,mode='min')
max_features = X_train.shape[1]
model = Sequential()



# input layer

model.add(Dense(50,  activation='relu'))

model.add(Dropout(0.5))



# hidden layer

model.add(Dense(25, activation='relu'))

model.add(Dropout(0.5))



# hidden layer

model.add(Dense(10, activation='relu'))

model.add(Dropout(0.5))



# output layer

model.add(Dense(units=1,activation='sigmoid'))



# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, 

          y=y_train, 

          epochs=80,

          validation_data=(X_test, y_test),

          #callbacks=Estopper

         )
loss=pd.DataFrame(model.history.history)
loss.plot()
predictions=model.predict_classes(test_data)
predictions=predictions.ravel()

predictions

test_data1=pd.read_csv('/kaggle/input/titanic/test.csv')

test_data1.head()
output=pd.DataFrame({'PassengerId':test_data1['PassengerId'],'Survived':predictions})

output.to_csv('my_submission.csv',index=False)