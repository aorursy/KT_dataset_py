import numpy as np

import pandas as pd

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data=pd.read_csv("/kaggle/input/reducing-commercial-aviation-fatalities/train.csv")

print(train_data.info())
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

import gc 

import warnings



warnings.filterwarnings('ignore')

scaler = MinMaxScaler()

trainN = train_data.loc[:, train_data.dtypes == np.float64]

trainN['seat'] = train_data.seat

trainN['crew'] = train_data.crew

trainN[:] = scaler.fit_transform(trainN[:])

trainN['experiment'] = train_data['experiment'].map({'CA': -1, 'DA': 0,'SS':1})

trainA=trainN[train_data.event=='A']

trainB=trainN[train_data.event=='B']

trainC=trainN[train_data.event=='C']

trainD=trainN[train_data.event=='D']



fig = plt.figure(figsize=(65,65))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

plt.grid()



for row,i in zip(trainN,range(0,len(trainN.columns))):

    

    plt.subplot(len(trainN.columns)/3, 4, i+1)

    plt.hist(trainA[row],label='A',alpha=0.4)

    plt.hist(trainB[row],label='B',alpha=0.4)

    plt.hist(trainC[row],label='C',alpha=0.4)

    plt.hist(trainD[row],label='D',alpha=0.4)

    plt.xlabel(row,size=26)

    plt.legend(fontsize=26)
fig = plt.figure(figsize=(55,25))

#fig.subplots_adjust(hspace=0.4, wspace=0.4)

plt.subplot(2, 2, 1)

corr = trainA.corr()



a = sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

plt.tick_params(axis='y', which='major', labelsize=26)

plt.tick_params(axis='x', labelrotation = 90,which='major', labelsize=26)



plt.subplot(2, 2, 2)

corr = trainB.corr()

b = sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

plt.tick_params(axis='y', which='major', labelsize=26)

plt.tick_params(axis='x', labelrotation = 90,which='major', labelsize=26)

plt.subplot(2, 2, 3)

corr = trainC.corr()

c = sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

plt.tick_params(axis='y', which='major', labelsize=26)

plt.tick_params(axis='x', labelrotation = 90,which='major', labelsize=26)

plt.subplot(2, 2, 4)

corr = trainD.corr()

d = sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

plt.tick_params(axis='y', which='major', labelsize=26)

plt.tick_params(axis='x', labelrotation = 90,which='major', labelsize=26)

 
crews = np.unique(train_data.crew)

grCrews = []

for c in crews:

    grCrews.append(trainN[train_data.crew==c])

fig = plt.figure(figsize=(65,65))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

plt.grid()

for row,i in zip(trainN,range(0,len(trainN.columns))):

    

    plt.subplot(len(trainN.columns)/3, 4, i+1)

    for gr,l in zip(grCrews,np.unique(train_data.crew)):

        plt.hist(gr[row].values,label=str(l),alpha=0.4)

    plt.xlabel(row,size=26)

    plt.legend(fontsize=26)

cov_mat = np.cov(trainN.values.T)

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)



tot = sum(eigen_vals)

var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)



plt.bar(range(1,len(trainN.columns)+1), var_exp, alpha=0.5, align='center',label='individual explained variance')

plt.step(range(1,len(trainN.columns)+1), cum_var_exp, where='mid',label='cumulative explained variance')



plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.grid()

plt.show()

from sklearn.decomposition import PCA

pca = PCA(0.95)

pca.fit(trainN)

df_pca=pca.transform(trainN)
y = train_data['event'].map({'A': 1, 'B': 2,'C':3,'D':4})
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(df_pca, y, test_size = 0.2, random_state = 66)
from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression(solver = 'lbfgs')

logisticRegr.fit(xTrain, yTrain)

y_pred=logisticRegr.predict(xTest)

logisticRegr.score(xTest,yTest)
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(yTest, y_pred)

print(confusion_matrix)

print(classification_report(yTest, y_pred))