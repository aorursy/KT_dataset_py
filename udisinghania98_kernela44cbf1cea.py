# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')



test = pd.read_csv('../input/test.csv')



sample = pd.read_csv('../input/gender_submission.csv')
train.head()
train.info()
test.info()
train[train.select_dtypes(include=['int64', 'float64']).columns] = train[train.select_dtypes(include=['int64', 'float64']).columns].apply(lambda x:x.fillna(x.median()))
test[test.select_dtypes(include=['int64', 'float64']).columns] = test[test.select_dtypes(include=['int64', 'float64']).columns].apply(lambda x:x.fillna(x.median()))
train.info()
test.info()
test = pd.merge(test, sample, left_on='PassengerId', right_on='PassengerId', how='inner')
test.info()
train.fillna(value=pd.np.nan, inplace=True)
test.fillna(value=pd.np.nan, inplace=True)
train = train.drop(['Name','Ticket','Cabin'], axis=1)

test = test.drop(['Name','Ticket','Cabin'], axis=1)
tr = pd.get_dummies(train, columns=train.select_dtypes('object').columns)

te = pd.get_dummies(test, columns=test.select_dtypes('object').columns)
tr.info()
te.info()
tr['Age'] = tr['Age'].astype('int')

tr['Fare'] = tr['Fare'].astype('int')

te['Age'] = te['Age'].astype('int')

te['Fare'] = te['Fare'].astype('int')
tr.info()
te.info()
y = tr.loc[:,'Survived']

X = tr.drop('Survived', axis=1)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from keras import Sequential

from keras.layers import Dense
len(X_train[0])
classifier = Sequential()



classifier.add(Dense(11,activation = 'relu', kernel_initializer = 'random_normal', input_dim=11))

classifier.add(Dense(7,activation = 'relu', kernel_initializer = 'random_normal'))

classifier.add(Dense(4,activation = 'relu', kernel_initializer = 'random_normal'))

classifier.add(Dense(1,activation = 'sigmoid', kernel_initializer = 'random_normal'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train, batch_size=10, epochs=50, verbose=2)
eval_model=classifier.evaluate(X_train, y_train)

eval_model
y_pred=classifier.predict(X_test)
y_pred
for i in y_pred:

    if i[0]>0.5:

        i[0]=1

    else:

        i[0]=0
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
y_train.head()
y1 = te.loc[:,'Survived']

X1 = te.drop('Survived', axis=1)

X1 = sc.fit_transform(X1)

X1
len(X1)
y_predictions=classifier.predict(X1)
q=[]

for i in y_predictions:

    if i[0]>0.5:

        q.append(float(1))

    else:

        q.append(float(0))
my_submission = pd.DataFrame({'PassengerId': sample.PassengerId, 'Survived': q})

# you could use any filename. We choose submission here

my_submission.to_csv('./submission.csv', index=False)
!ls