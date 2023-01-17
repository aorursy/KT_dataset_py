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
!dir ../input/male-daan-schnell-mal-klassifizieren
fn = '../input/male-daan-schnell-mal-klassifizieren/train.csv'

import pandas as pd

df = pd.read_csv(fn)

df.head(5)
df = pd.read_csv(fn,index_col='Id')

df.head()
data = df.values

data
data.shape,data.dtype
nTrain=5000

Xtrain = data[:nTrain,:-1] 

ytrain = data[:nTrain,-1] 

Xtest = data[nTrain:,:-1] 

ytest = data[nTrain:,-1] 

Xtrain.shape,ytrain.shape,Xtest.shape,ytest.shape
ytrain
ytrain = ytrain.astype('int')

ytest = ytest.astype('int')

ytrain
k=7
#np.random.randint? #Tipp: entkommentieren Sie diese Zeile und lernen Sie, was die Funktion randint ist.

zufälliger_Index = np.random.randint(low=0,high=len(ytest))

zufälliger_Index
Testzeile = Xtest[zufälliger_Index,:]

Testlabel = ytest[zufälliger_Index]

Testzeile,Testlabel
distanz = (((Xtrain - Testzeile)**2).sum(axis=1))**0.5
distanz
a= np.array([1,4,8,2])

sorted_indices = np.argsort(a)

sorted_indices, a[sorted_indices]

sorted_indices = np.argsort(distanz)

distanz[sorted_indices]
distanz[sorted_indices[:k]], sorted_indices[:k], ytrain[sorted_indices[:k]]
Auftretende_Trainingslabels = ytrain[sorted_indices[:k]]
if np.mean(Auftretende_Trainingslabels)>=0.5:

    yhat = 1

else:

    yhat = 0

yhat
print('Häufigstes Label in [1,0,1,0,0]:',np.argmax(np.bincount([1,0,1,0,0])))

print('Häufigstes Label in [1,0,1,0,1]:',np.argmax(np.bincount([1,0,1,0,1])))
def kNN_Vorhersage(Xtrain,Testzeile,k):

    distanz = (((Xtrain - Testzeile)**2).sum(axis=1))**0.5

    sorted_indices = np.argsort(distanz)

    Auftretende_Trainingslabels = ytrain[sorted_indices[:k]]

    return np.argmax(np.bincount(Auftretende_Trainingslabels))



zufälliger_Index = np.random.randint(low=0,high=len(ytest))

Testzeile = Xtest[zufälliger_Index,:]



yhat = kNN_Vorhersage(Xtrain,Testzeile,k)



print(f'Vorhersage für {Testzeile}:{yhat}')
#kNN-Klassifikator in Scikit-Learn

from sklearn.neighbors import KNeighborsClassifier 

clf = KNeighborsClassifier(n_neighbors=k)

#KNeighborsClassifier? #Bitte entkommentieren!
#Laden der Daten:

import pandas as pd

df_train = pd.read_csv('../input/male-daan-schnell-mal-klassifizieren/train.csv',index_col='Id')

df_test = pd.read_csv('../input/male-daan-schnell-mal-klassifizieren/test.csv',index_col='Id')

df_train.head(3)
# Bringe Daten in X-y-Form:

Xtrain = df_train.values[:,:-1]

ytrain = df_train.values[:,-1]

Xtest = df_test.values
#Trainiere den k-NN:

clf.fit(Xtrain,ytrain)
yhat = clf.predict(Xtest)
#Erstelle eine .csv-Datei "Submission.csv"

ser = pd.Series(yhat,name='y').astype('int')

ser.index.name='Id'

ser.to_csv('Submission.csv',header=True)

!head Submission.csv