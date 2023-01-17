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
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow import keras
filepath= '/kaggle/input/titanic/train.csv'

traindata= pd.read_csv(filepath)

traindata = traindata[['PassengerId','Age','Pclass','Name','Sex','SibSp','Parch','Ticket','Cabin','Embarked','Survived']]

traindata.head()

#We need to tend to the missing values in the age column. For this, instead of dropping the entire column, we fill the null values by the mean of all the ages

traindata['Age'].fillna((traindata['Age'].mean()), inplace=True)  

#We convert the names column to just the honorofic to check if it influences the survival rate

col_one_list = traindata['Name'].tolist()

p=[]

for a in col_one_list:

    b=a.split(' ')

    #print(b)

    #p.append(b[1])

    if b[1]=='Mr.' or b[1]=='Mrs.' or b[1]=='Miss.' or b[1]=='Master.':

          p.append(b[1])

    else:

          p.append('rare')

            

traindata['honorifics'] = p

one_hot = pd.get_dummies(traindata['honorifics'])

traindata = traindata.drop('honorifics',axis = 1)

traindata = traindata.join(one_hot)

traindata=traindata.drop('Cabin',axis=1)

#We also remove the Ticket number, passenger ID and the Name columns because they are irrelavant to our model



traindata=traindata.drop('PassengerId',axis=1)

traindata=traindata.drop('Name',axis=1)

traindata=traindata.drop('Ticket',axis=1)

#Perform one hot encoding on the Pclass column

one_hot = pd.get_dummies(traindata['Pclass'])

# Drop column Product as it is now encoded

traindata = traindata.drop('Pclass',axis = 1)

# Join the encoded df

traindata = traindata.join(one_hot)

# Similarly, perform one hot encoding on the Embarked column

one_hot = pd.get_dummies(traindata['Embarked'])

traindata = traindata.drop('Embarked',axis = 1)

traindata = traindata.join(one_hot)

#Change the categorical variables in the Sex column to numbers

traindata['Sex'] = traindata['Sex'].replace('male', 0)

traindata['Sex'] = traindata['Sex'].replace('female', 1)

traindata.head()

y=traindata['Survived']

x=traindata.drop('Survived',axis=1)

#Splitting training and testing data

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)
x_train
traindata.shape
model=keras.Sequential([

    keras.layers.Dense(10,input_shape=(15,)),

    keras.layers.Dense(5,activation=tf.nn.relu),

    keras.layers.Dense(2,activation="softmax")

    

])
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['acc']) 
prediction=model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))
#Now we preprocess the test data exactly like we did the training data.



filepath= '/kaggle/input/titanic/test.csv'

testdata= pd.read_csv(filepath)

testdata = testdata[['PassengerId','Age','Pclass','Name','Sex','SibSp','Parch','Ticket','Cabin','Embarked']]



col_one_list = testdata['Name'].tolist() 

p=[]

for a in col_one_list:

    b=a.split(' ')

    #print(b)

    #p.append(b[1])

    if b[1]=='Mr.' or b[1]=='Mrs.' or b[1]=='Miss.' or b[1]=='Master.':

          p.append(b[1])

    else:

          p.append('rare')

            

testdata['honorifics'] = p

testdata=testdata.drop('Cabin',axis=1)

testdata=testdata.drop('PassengerId',axis=1)

testdata=testdata.drop('Name',axis=1)

testdata=testdata.drop('Ticket',axis=1)

testdata['Age'].fillna((testdata['Age'].mean()), inplace=True)

one_hot = pd.get_dummies(testdata['honorifics'])

testdata = testdata.drop('honorifics',axis = 1)

testdata = testdata.join(one_hot)



one_hot = pd.get_dummies(testdata['Pclass'])

testdata = testdata.drop('Pclass',axis = 1)

testdata = testdata.join(one_hot)

one_hot = pd.get_dummies(testdata['Embarked'])

testdata = testdata.drop('Embarked',axis = 1)

testdata = testdata.join(one_hot)

testdata.head()

testdata['Sex'] = testdata['Sex'].replace('male', 0)

testdata['Sex'] = testdata['Sex'].replace('female', 1)

testdata.head()
prediction= model.predict(testdata)
prediction
a=prediction.tolist()

pred=[]

for i in a:

   # print(i[0])

    if i[0]>i[1]:

        pred.append(0)

    else:

        pred.append(1)
pred
submissiondata=pd.read_csv('/kaggle/input/titanic/test.csv')

Passengerid=submissiondata['PassengerId'].tolist() 

output=pd.DataFrame(list(zip(Passengerid, pred)),

              columns=['PassengerId','Survived'])

output.head()

output.to_csv('my_submission.csv', index=False)
output