import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from datetime import datetime

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

import sklearn.tree

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPRegressor

data = pd.read_csv('/kaggle/input/uncover/apple_mobility_trends/mobility-trends.csv')
Paises=np.unique(np.array(data['region']))

Transportes=['driving','walking']

for P in Paises[[1,15,25,29,54,60,110]]:

    plt.figure(figsize=(15,4))

    for i,T in enumerate(Transportes): 

        dataD=data[data['region']==P]    

        dataDr=dataD[dataD['transportation_type']==T]

        plt.subplot(1,2,i+1)

        plt.plot(np.array(range(dataDr.shape[0])),dataDr['value'])

        plt.title(P+'  '+T)

        plt.xlabel("days")

        plt.ylabel("Percentage comparison with day zero")

    plt.show()
regresion = sklearn.linear_model.LinearRegression()

ColD=data[data['region']=='Colombia']

Scores=np.zeros((2,Paises.shape[0]))

count=0

for T in Transportes:

    for i,P in enumerate(Paises):

        Data=data[data['region']==P]    

        Datadr=Data[Data['transportation_type']==T]

        ColDr=ColD[ColD['transportation_type']==T]

        regresion.fit(np.array(range(Datadr.shape[0])).reshape(-1,1),Datadr['value'])

        Scores[count,i]=(regresion.score(np.array(range(ColDr.shape[0])).reshape(-1,1),ColDr['value']))

    count=count+1

plt.figure(figsize=(30,30))

plt.scatter(Scores[0],Scores[1])

plt.scatter(Scores[0,29],Scores[1,29],s=300)

for i,P in enumerate(Paises):

    plt.annotate(P, (Scores[0,i], Scores[1,i]),size=20)

plt.xlabel("Colombia driving Score")

plt.ylabel("Colombia walking Score")

plt.show()
v1=Scores[0]>0.5

v2=Scores[1]>0.5

NScores=Scores[:,v1*v2]

NPaises=Paises[v1*v2]

plt.figure(figsize=(30,30))

plt.scatter(NScores[0],NScores[1])

plt.scatter(Scores[0,29],Scores[1,29],s=800)

for i,P in enumerate(NPaises):

    plt.annotate(P, (NScores[0,i], NScores[1,i]),size=20)

plt.xlabel("Colombia driving Score")

plt.ylabel("Colombia walking Score")

plt.show()
Distances=np.zeros(152)

for i in range(152):

    Distances[i]=np.sqrt((Scores[0,i]-Scores[0,29])**2+(Scores[1,i]-Scores[1,29])**2)

args=np.argsort(Distances)

argsi=np.argsort(-Distances)

for P in Paises[args[0:3]]:

    plt.figure(figsize=(15,4))

    for i,T in enumerate(Transportes): 

        dataD=data[data['region']==P]    

        dataDr=dataD[dataD['transportation_type']==T]

        plt.subplot(1,2,i+1)

        plt.plot(np.array(range(dataDr.shape[0])),dataDr['value'])

        plt.title(P+'  '+T)

        plt.xlabel("days")

        plt.ylabel("Percentage comparison with day zero")

    plt.show()
for P in Paises[argsi[0:3]]:

    plt.figure(figsize=(15,4))

    for i,T in enumerate(Transportes): 

        dataD=data[data['region']==P]    

        dataDr=dataD[dataD['transportation_type']==T]

        plt.subplot(1,2,i+1)

        plt.plot(np.array(range(dataDr.shape[0])),dataDr['value'])

        plt.title(P+'  '+T)

        plt.xlabel("days")

        plt.ylabel("Percentage comparison with day zero")

    plt.show()