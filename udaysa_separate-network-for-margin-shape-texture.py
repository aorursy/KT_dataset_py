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
# import required libraries 

%pylab inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

# from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import LabelEncoder



from keras.models import Sequential, Merge

from keras.layers import Dense,Dropout,Activation

from keras.utils.np_utils import to_categorical

#from keras.callbacks import EarlyStopping
data = pd.DataFrame.from_csv('../input/train.csv')

y = data['species']

y = LabelEncoder().fit(y).transform(y)

y_cat = to_categorical(y)



margin = data.columns[1:65]

margin = data[margin].as_matrix()

margin = StandardScaler().fit(margin).transform(margin)

shape = data.columns[65:129]

shape = data[shape].as_matrix()

shape = StandardScaler().fit(shape).transform(shape)

texture = data.columns[129:193]

texture = data[texture].as_matrix()

texture = StandardScaler().fit(texture).transform(texture)

# Define separate model for each meta feature and its 64 values 

modelMargin = Sequential()

modelMargin.add(Dense(128, input_dim=64, activation='relu'))

modelMargin.add(Dropout(0.7))



modelShape = Sequential()

modelShape.add(Dense(128, input_dim=64, activation='relu'))

modelShape.add(Dropout(0.7))



modelTexture = Sequential()

modelTexture.add(Dense(128, input_dim=64, activation='relu'))

modelTexture.add(Dropout(0.7))



# merge all models

merged = Merge([modelMargin, modelShape, modelTexture], mode='concat')
model = Sequential()

model.add(merged)

model.add(Dense(99, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(

    [margin, shape, texture], 

    y_cat, 

    nb_epoch=350,

    batch_size=32,

    validation_split=0.1,

    verbose=0

)
# summarize history for loss

## Plotting the loss with the number of iterations



plt.semilogy(history.history['loss'])

plt.semilogy(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
## read test file

test1 = pd.read_csv('../input/test.csv')

test = pd.DataFrame.from_csv('../input/test.csv')

index = test1.pop('id')

# index = test['id']



testMargin = test[test.columns[0:64]].as_matrix()

testShape = test[test.columns[64:128]].as_matrix()

testTexture = test[test.columns[128:192]].as_matrix()



testMargin = StandardScaler().fit(testMargin).transform(testMargin)

testShape = StandardScaler().fit(testShape).transform(testShape)

testTexture = StandardScaler().fit(testTexture).transform(testTexture)



yPred = model.predict_proba(

    [testMargin, testShape, testTexture]

)



# ## Converting the test predictions in a dataframe as depicted by sample submission

columns = data['species'].unique()

yPred = pd.DataFrame(yPred, index=index, columns=sort(columns))

fp = open('merged_nn2.csv','w')

fp.write(yPred.to_csv())