# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.






# create data frame containing your data, each column can be accessed # by df['column   name']

dataset = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')

del dataset['nameDest']

del dataset['nameOrig']

del dataset['type']



dataset.head()
#Exploratory Data Analysis 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.figure(figsize=(12,8))

sns.pairplot(dataset[['amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFraud']], hue='isFraud')
#Splitting the Training/Test Data



from sklearn.model_selection  import train_test_split

X, y = dataset.iloc[:,:-2], dataset.iloc[:, -2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from keras.models import Sequential



model = Sequential()



from keras.layers import Dense, Activation



input_dim = X_train.shape[1]

nb_classes = y_train.shape[0]



model.add(Dense(units=1, input_dim=input_dim))

model.add(Activation('relu'))

model.add(Dense(units=1))

model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy',

              optimizer='sgd',

              metrics=['accuracy'])
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.

#model.fit(X_train, y_train, epochs=5, batch_size=32)

#y_train_binary = keras.utils.to_categorical(y_train[:0])

model.fit(X_train.as_matrix(), y_train[0:].as_matrix(), epochs=1, batch_size=500)
score = model.evaluate(X_test.as_matrix(), y_test[0:].as_matrix(), batch_size=500)

y