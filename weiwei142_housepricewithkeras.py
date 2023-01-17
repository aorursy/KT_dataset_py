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
data = pd.read_csv('../input/train.csv')
data.head()
import matplotlib.pyplot as plt

import seaborn as sns

k = 10

cols = data.corr().nlargest(k,'SalePrice')['SalePrice'].index

sns.set(font_scale=1.25)

fig, ax = plt.subplots(figsize=(20,15))

sns.heatmap(data[cols].corr(),annot=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,ax=ax)
sns.set()

cols = ['SalePrice','OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

sns.pairplot(data[cols], size = 2.5)

plt.show()


cols = ['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

x = data[cols].values

y = data['SalePrice'].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn import preprocessing



x_train = preprocessing.StandardScaler().fit_transform(x_train)

#y_train = preprocessing.StandardScaler().fit_transform(y_train.reshape(-1,1))



x_test = preprocessing.StandardScaler().fit_transform(x_test)

#y_test = preprocessing.StandardScaler().fit_transform(y_test.reshape(-1,1))
from keras.models import Sequential

from keras.layers import Dense,Dropout



def build_model():

    model = Sequential()

    model.add(Dense(128,activation='relu',input_shape=(x_train.shape[1],)))

    model.add(Dense(64,activation='relu'))

    model.add(Dense(32,activation='relu'))

    model.add(Dense(16,activation='relu'))

    model.add(Dense(8,activation='relu'))

    model.add(Dense(4,activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

    return model
model = build_model()

history = model.fit(x_train,y_train,epochs=100,batch_size=32,validation_data=(x_test,y_test))

import matplotlib.pyplot as plt

def show(history,loss,val_loss,label):

    epochs = range(1,len(history.history[loss])+1)

    plt.plot(epochs,history.history[loss],label=label)

    plt.plot(epochs,history.history[val_loss],label='Validation '+label)

    plt.title(label)

    plt.legend()

plt.figure(figsize=(25,8))

plt.subplot(121)

show(history,'mean_absolute_error','val_mean_absolute_error','mean_absolute_error')

plt.subplot(122)

show(history,'loss','val_loss','loss')



plt.show()
test_origin = pd.read_csv('../input/test.csv')

test = test_origin[cols]

#Fill the null data

test['GarageCars'].fillna(1.766118, inplace=True)

test['TotalBsmtSF'].fillna(1046.117970, inplace=True)

test = preprocessing.StandardScaler().fit_transform(test)

pred = model.predict(test)



origin_pred = pd.DataFrame(pred,columns=['SalePrice'])

result = pd.concat([test_origin['Id'], origin_pred], axis=1)

result
result.to_csv('./Predictions.csv', index=False)