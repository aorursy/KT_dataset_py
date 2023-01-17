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
import matplotlib.pyplot as plt

import seaborn as sns
filePath='/kaggle/input/boston-house-prices/housing.csv'
names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

df=pd.read_csv(filePath,delim_whitespace=True,header=None,names=names)

df.head(10)
df.info()
df.describe().T
corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corrmat, vmax=1, square=True,annot=True,center=0,cmap=cmap)

plt.show()
numpy_data=np.genfromtxt(filePath)

data=numpy_data[:,0:13]

target=numpy_data[:,13]
print(data.shape)

print(target.shape)
import matplotlib.pyplot as plt





def drawScatter(x):

    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

    plt.scatter(df[x], df['MEDV'])

    #plt.title(x+' & MEDV')

    plt.xlabel(x)

    plt.ylabel('MEDV')

    plt.grid()

    plt.show()
drawScatter('CRIM')
drawScatter('ZN')
drawScatter('INDUS')
drawScatter('CHAS')
drawScatter('NOX')
drawScatter('RM')
drawScatter('AGE')
drawScatter('DIS')
drawScatter('RAD')
drawScatter('TAX')
drawScatter('PTRATIO')
drawScatter('B')
drawScatter('LSTAT')
from sklearn.model_selection import train_test_split


X=df[names[0:-1]]

Y=df[names[-1]]



x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size=0.3)

print(x_train.shape)

print(x_test.shape)
from sklearn.ensemble import GradientBoostingRegressor
gdbt=GradientBoostingRegressor(loss='ls')

gdbt.fit(x_train, y_train)

gdbt.score(x_test, y_test)



gdbt_y_predict=gdbt.predict(x_test)
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error 

from sklearn.metrics import mean_absolute_error 
print('r2', r2_score(y_test,gdbt_y_predict))

print('mse',mean_squared_error(y_test,gdbt_y_predict))

print('mae',mean_absolute_error(y_test,gdbt_y_predict))

print('rmse',np.sqrt(mean_squared_error(y_test,gdbt_y_predict)))
plt.grid()

plt.scatter(range(1,len(x_test)+1),gdbt_y_predict,label='predict' )

plt.scatter(range(1,len(x_test)+1),y_test ,label='actual')

plt.legend()

plt.xlabel('index')

plt.ylabel('price')



plt.show()
mean=x_train.mean(axis=0)

std=x_train.std(axis=0)

train_data_processed=(x_train-mean)/std



test_data_processed=(x_test-mean)/std



print(train_data_processed.mean(axis=0))

print(train_data_processed.std(axis=0))
from keras import models

from keras import layers
def build_model():

  model=models.Sequential()

  model.add(layers.Dense(64,activation='relu',input_shape=(train_data_processed.shape[1],)))

  model.add(layers.Dense(64,activation='relu'))

  model.add(layers.Dense(1))

  model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

  return model
model = build_model()

history=model.fit(train_data_processed, y_train,epochs=80, batch_size=16, verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data_processed, y_test)



import matplotlib.pyplot as plt

plt.plot(range(1, len(history.history['mae']) + 1), history.history['mae'])

plt.xlabel('Epochs')

plt.ylabel('Validation MAE')

plt.show()
print(test_mse_score)

print(test_mae_score)
predict_result=model.predict(test_data_processed)
print('r2', r2_score(y_test,predict_result))

print('mse',mean_squared_error(y_test,predict_result))

print('mae',mean_absolute_error(y_test,predict_result))

print('rmse',np.sqrt(mean_squared_error(y_test,predict_result)))
plt.grid()

plt.scatter(range(1,predict_result.shape[0]+1),predict_result,label='predict' )

plt.scatter(range(1,predict_result.shape[0]+1),y_test ,label='actual')

plt.scatter(range(1,len(x_test)+1),gdbt_y_predict,label='gbdt_predict' )

plt.legend()

plt.xlabel('index')

plt.ylabel('price')



plt.show()