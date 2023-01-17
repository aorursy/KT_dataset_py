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
df1=pd.read_csv('/kaggle/input/padhai-mp-neuron-like-unlike-classification/train.csv')

n_rows,n_columns=df1.shape



print(f'There are {n_rows} number of rows and {n_columns} columns')
df1.head(5)
df1.groupby('Rating')['Brand'].count()
df1.fillna('No value',inplace=True)
df1.isnull().sum()
from sklearn.preprocessing import LabelEncoder



feature=['Brand','Capacity','Bluetooth','Camera Features','Chipset','Colours','Display Type','Expandable Memory','Fingerprint Sensor','Flash','GPS','Graphics','Height','Image Resolution','Internal Memory','Loudspeaker','Model','Network','Operating System','Other Sensors','Pixel Density','Processor','RAM','Rating Count','Resolution','Review Count','SIM 1','SIM Size','SIM Slot(s)','Screen Resolution','Screen Size','Screen to Body Ratio (calculated)','Thickness','Touch Screen','Type','USB Connectivity','User Replaceable','Weight','Wi-Fi','Wi-Fi Features','Width']



encoder=LabelEncoder()

encoded=df1[feature].apply(encoder.fit_transform)

encoded.head(10)
from sklearn.model_selection import train_test_split

X=encoded

Y=df1['Rating']



X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.9)

print(X.shape,X_train.shape,X_test.shape)

print(Y.mean(),Y_train.mean(),Y_test.mean())
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.9,random_state=2)
print(Y_train,Y_test)
import matplotlib.pyplot as plt

plt.plot(X_train,'*',)

plt.show()

import matplotlib.pyplot as plt

plt.plot(Y_train,'*',)

plt.show()

plt.plot(X_train.T,'*')

plt.xticks(rotation='vertical')

plt.show()

type(X_train)
X_binarised_3_train=X_train['Model'].map(lambda x:0 if x<175 else 1)

plt.plot(X_binarised_3_train,'*')

type(X_binarised_3_train)
plt.plot(Y_train.T,'*')

plt.xticks(rotation='vertical')

plt.show()

type(Y_train)
Y_binarised_train=Y_train.map(lambda x:0 if x<4 else 1)

plt.plot(Y_binarised_train,'*')

type(Y_binarised_train)
X_binarised_train=X_train.apply(pd.cut,bins=2,labels=[1,0])

type(X_binarised_train)

plt.plot(X_binarised_train,'*')
plt.plot(X_binarised_train.T,'*')

plt.xticks(rotation='vertical')

plt.show()
plt.plot(Y_binarised_train.T,'*')

plt.xticks(rotation='vertical')

plt.show()
X_binarised_test=X_test.apply(pd.cut,bins=2,labels=[1,0])

type(X_binarised_test)

plt.plot(X_binarised_test,'*')
plt.plot(X_binarised_test.T,'*')

plt.xticks(rotation='vertical')

plt.show()

type(X_binarised_test)
Y_binarised_test=Y_test.map(lambda x:0 if x<4 else 1)

plt.plot(Y_binarised_test,'*')

type(Y_binarised_test)
X_binarised_test=X_binarised_test.values

X_binarised_train=X_binarised_train.values

(type(X_binarised_test))

Y_binarised_test=Y_binarised_test.values

Y_binarised_train=Y_binarised_train.values

(type(Y_binarised_test))

from random import randint

b=3

i=randint(0,X_binarised_train.shape[0])

print('for row',i)

if (np.sum(X_binarised_train[100, :])>=b):

    print("mp neuron inference liked")

else:

    print("mp neuron inference not liked")



if (Y_binarised_train[i]==1):

    print('Ground truth is liked by people')

else:

    print('Ground truth is not liked by people')

    



    
b=3



Y_pred_train=[]

accurate_rows=0

for x,y in zip(X_binarised_train,Y_binarised_train):

    Y_pred=(np.sum(x) >= b)

    Y_pred_train.append(Y_pred)

    accurate_rows += (y==Y_pred)

print(accurate_rows,accurate_rows/X_binarised_train.shape[0])
for b in range(X_binarised_train.shape[1]+1):

    Y_pred_train=[]

    accurate_rows=0

    for x,y in zip(X_binarised_train,Y_binarised_train):

        Y_pred=(np.sum(x) >= b)

        Y_pred_train.append(Y_pred)

        accurate_rows += (y==Y_pred)

    print(b,accurate_rows/X_binarised_train.shape[0])
from sklearn.metrics import accuracy_score

Y_pred_test=[]



b=14



for x in X_binarised_test:

    Y_pred=(np.sum(x) >= b)

    Y_pred_test.append(Y_pred)

accuracy=accuracy_score(Y_pred_test,Y_binarised_test)

print(b,accuracy)
class MPNeuron:

    def __init__(self):

        self.b=None # it is based on number of parameter in the model

        

    def model(self,x):

        return(sum(x) >= self.b)

    

    def predict(self,X):

        Y=[]

        for x in X:

            result=self.model(x)

            Y.append(result)

        return np.array(Y)

    def fit(self,X,Y):

        accuracy={}

        

        for b in range(X.shape[1] +1):

            self.b=b

            Y_pred=self.predict(X)

            accuracy[b]=accuracy_score(Y_pred,Y)

        best_b=max(accuracy, key = accuracy.get)

        self.b=best_b

        

        print('optimal value:',best_b)

        print('Highest accuracy',accuracy[best_b])

        
mpneuron=MPNeuron()

mpneuron.fit(X_binarised_train,Y_binarised_train)
y_test_pred=mpneuron.predict(X_binarised_test)

accuracy_test=accuracy_score(y_test_pred,Y_binarised_test)
print(accuracy_test)