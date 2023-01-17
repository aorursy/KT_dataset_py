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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



df = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv",na_values =['?'])

df.dropna(inplace = True)





X = df.iloc[:, 1:-1].values

y = df.iloc[:, -1].values
# Encoding categorical data



from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.compose import ColumnTransformer



labelencoder_X = LabelEncoder()

X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

transformer = ColumnTransformer(

    transformers=[

        ("OneHot",        # Just a name

         OneHotEncoder(), # The transformer class

         [2]              # The column(s) to be applied on.

         )

    ],

    remainder='passthrough' # donot apply anything to the remaining columns

)

X = transformer.fit_transform(X.tolist())

X = X.astype('float64')



from keras.utils import np_utils

y = np_utils.to_categorical(y)



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras import layers

from keras import models





from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.regularizers import l2

from keras.layers import BatchNormalization

from keras import optimizers

# Initialising the ANN

model = Sequential()



model.add(Dense(80,input_dim=13,kernel_initializer = 'uniform', activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))



model.add(BatchNormalization())



model.add(Dense(40,kernel_initializer = 'uniform', activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))



model.add(BatchNormalization())



model.add(Dense(20,kernel_initializer = 'uniform', activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))



model.add(BatchNormalization()) 



model.add(Dense(10,kernel_initializer = 'uniform', activation = 'relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))



model.add(Dense(6,kernel_initializer = 'uniform',activation='softmax'))



adam = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)



model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

history = model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size = 20, epochs = 100)



#Predicting on test set



df_test = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv",na_values =['?'])

ids = df_test.iloc[:,0]

X_predict = df_test.iloc[:, 1:].values

labelencoder = LabelEncoder()

X_predict[:, 2] = labelencoder.fit_transform(X_predict[:, 2])

transformer = ColumnTransformer(

    transformers=[

        ("OneHot",        # Just a name

         OneHotEncoder(), # The transformer class.

         [2]              # The column(s) to be applied on.

         )

    ],

    remainder='passthrough' # donot apply anything to the remaining columns

)

X_predict = transformer.fit_transform(X_predict.tolist())

X_predict = X_predict.astype('float64')
y_predict = model.predict_classes(X_predict)
u3,c3 = np.unique(y_predict, return_counts=True)

u4,c4 = np.unique(y.argmax(1), return_counts=True)

print(np.asarray((u4, c4*100/sum(c4))))

print(np.asarray((u3, c3*100/sum(c3))))

xyz = pd.DataFrame({"ID" : ids, "Class" : y_predict})

xyz.to_csv("result.csv", index=False)
df2 = pd.read_csv('/kaggle/working/result.csv')

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df2)