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
seed=7

np.random.seed(seed)
# step 1: load dataset

dataset=pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

print(dataset.head())

print(dataset.info())# No missing values, No categorical variables are present
X=dataset.loc[:,'Pregnancies':'Age'] 

Y=dataset.loc[:,'Outcome']
from keras.models import Sequential

from keras.layers import Dense
#Dense = no. of hidden layer( i have kept 12 in this case)

#input_dim=no. of input 

# init :we initialize the network weights to a small random number generated from a uniform distribution (uniform),

# in this case between 0 and 0.05 because that is the default uniform weight initialization in Keras.

# Another traditional alternative would be normal for small random numbers generated from a Gaussian distribution. 

#sigmoid activation function on the output layer to ensure our network output is between 0 and 1

# relu works as below:

# f(x) : 0 if x<0

#      : x if x>=0 
model= Sequential()

model.add(Dense(12,input_dim=8,init='uniform',activation='relu')) 

model.add(Dense(8, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model

model.fit(X, Y,validation_split=0.33,nb_epoch=150, batch_size=10)

# evaluate the model

scores = model.evaluate(X, Y) 

scores
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))