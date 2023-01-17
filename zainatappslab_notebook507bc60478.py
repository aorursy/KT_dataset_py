# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense

from keras import optimizers



np.random.seed(7)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

df = pd.read_csv('../input/results.csv')



X        = df[list(set(df.columns) - set(['away_ft']))]

Y        = df[list(set(['away_ft']))]



X = pd.get_dummies(X[:10000])

Y = Y[:10000]



print(np.shape(X)[1])

model = Sequential()



model.add(Dense(np.shape(X)[1], kernel_initializer='uniform', input_dim=np.shape(X)[1], activation='relu'))

model.add(Dense(1000,  activation='relu'))

model.add(Dense(1, activation='softmax'))

sgd = optimizers.SGD(lr=0.9, decay=1e-6, momentum=0.9)

model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])



model.fit(np.array(X), np.array(Y), epochs=1, batch_size=1000)



scores = model.evaluate(np.array(X), np.array(Y))

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Any results you write to the current directory are saved as output.