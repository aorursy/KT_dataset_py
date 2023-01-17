from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
nRowsRead = 1000 # specify 'None' if want to read whole file

df1 = pd.read_csv('../input/Anlise de vagas de T.I.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'Planilha sem ttulo - Respostas ao formulrio 1.csv'

nRow, nCol = df1.shape

print(f'O dataset: {df1.dataframeName} possui {nRow} linhas e {nCol} colunas')
df1.head()
df1["A oferta de trabalho é referente a que área de T.I?"].value_counts()
df1["Qual(quais) conhecimento(s) de tecnologia está sendo exigido na oferta de trabalho?"].value_counts()
df1["Nesta oferta de trabalho, está sendo exigido certificação? Se sim, quais?"].value_counts()
df1["Nesta oferta de trabalho, está sendo exigido experiência?"].value_counts()
df1["Qual a média salarial proposta?"].value_counts()
df1["A oferta de trabalho é para qual região do país?"].value_counts()
df1["Para ocupar está vaga, é exigido ensino superior?"].value_counts()
df1["Está vaga é referente a que data?"].value_counts()
train = pd.read_csv("../input/Anlise de vagas de T.I.csv")
train = train.iloc[:,0:31]

train.head(10)
from sklearn.preprocessing import LabelEncoder 

labelencoder = LabelEncoder()

train['A oferta de trabalho é referente a que área de T.I?'] = labelencoder.fit_transform(train['A oferta de trabalho é referente a que área de T.I?'])

train.head()
x = train.drop('A oferta de trabalho é referente a que área de T.I?', axis=1)

y = train['A oferta de trabalho é referente a que área de T.I?']
x.head(5)
print(x.shape)

print(y.shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
from keras.models import Sequential

from keras.layers import Dense