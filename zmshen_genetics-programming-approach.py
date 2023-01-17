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
df = pd.read_csv('../input/train.csv')

df.head() #Liat 5 elemen teratas
df['id'] = df['Unnamed: 0'] #buat kolom baru namanya id

df = df.drop(columns='Unnamed: 0') #ilangin kolom Unnamed: 0

df = df.set_index(df['id'])

df = df.drop(columns='id') #ilangin kolom id

df.head()
df.info()
df.describe()
x = df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]

y = df['Loan_Status']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2) #Split train test
from gplearn.genetic import SymbolicClassifier

clf = SymbolicClassifier(parsimony_coefficient=.001,population_size=1000,

                         random_state=0)

train = clf.fit(x_train, y_train)

pred = train.predict(x_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, pred)
soal = pd.read_csv('../input/test.csv')

soal.head()
soal['id'] = soal['Unnamed: 0'] #buat kolom baru namanya id

soal = soal.drop(columns='Unnamed: 0') #ilangin kolom Unnamed: 0

soal = soal.set_index(soal['id'])

soal = soal.drop(columns='id') #ilangin kolom id

soal.head()
x_soal = soal[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

soal_scaled = scaler.fit_transform(x_soal)
jawab = train.predict(soal_scaled)
submission = pd.DataFrame({'Loan_Status' : jawab})

submission = submission.set_index(soal.index)

submission.head()
submission.to_csv('submission.csv')