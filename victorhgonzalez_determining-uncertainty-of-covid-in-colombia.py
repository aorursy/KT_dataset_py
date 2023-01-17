# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib.pyplot as plt

import sklearn.cluster

from matplotlib.lines import Line2D

import umap

import torch

import sklearn.neural_network



%matplotlib inline

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
eddc=pd.read_csv('/kaggle/input/uncover/UNCOVER/ECDC/current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv')

eddc=eddc.drop('daterep',axis='columns')

eddc['casesNorm']=eddc['cases']/eddc['popdata2018']

eddc['deathsNorm']=eddc['deaths']/eddc['popdata2018']

eddc=eddc.dropna(axis=0)

eddc=eddc.set_index(eddc['countriesandterritories'])
april4th=(eddc[(eddc['day']==28) & (eddc['month']==4)])

plt.figure(figsize=(14,7))



asia=april4th[april4th['continentexp']=='Asia']

america=april4th[april4th['continentexp']=='America']

europe=april4th[april4th['continentexp']=='Europe']

oceania=april4th[april4th['continentexp']=='Oceania']

africa=april4th[april4th['continentexp']=='Africa']



plt.subplot(1,2,1)



plt.plot(asia['casesNorm'],asia['deathsNorm'],'.',label='Asia',markersize=10,color='red')

plt.plot(america['casesNorm'],america['deathsNorm'],'.',label='America',markersize=10,color='blue')

plt.plot(europe['casesNorm'],europe['deathsNorm'],'.',label='Europe',markersize=10,color='green')

plt.plot(oceania['casesNorm'],oceania['deathsNorm'],'.',label='Oceania',markersize=10,color='black')

plt.plot(africa['casesNorm'],africa['deathsNorm'],'.',label='Africa',markersize=10,color='cyan')



#for index,row in april4th.iterrows():

#    plt.annotate(s=row['countriesandterritories'],xy=(row['cases'],row['deaths']))

plt.xlabel('Cases')

plt.ylabel('Deaths')

plt.legend()



plt.subplot(1,2,2)



plt.plot(asia['casesNorm'],asia['deathsNorm'],'.',label='Asia',markersize=10,color='red')

plt.plot(america['casesNorm'],america['deathsNorm'],'.',label='America',markersize=10,color='blue')

plt.plot(europe['casesNorm'],europe['deathsNorm'],'.',label='Europe',markersize=10,color='green')

plt.plot(oceania['casesNorm'],oceania['deathsNorm'],'.',label='Oceania',markersize=10,color='black')

plt.plot(africa['casesNorm'],africa['deathsNorm'],'.',label='Africa',markersize=10,color='cyan')

plt.title('Deaths and cases')

#for index,row in april4th.iterrows():

#    plt.annotate(s=row['countriesandterritories'],xy=(row['cases'],row['deaths']))

plt.xscale('log')

plt.yscale('log')

plt.xlabel('Cases per capita')

plt.ylabel('Deaths per capita')

plt.legend()
tsne = sklearn.manifold.TSNE(perplexity=100)

predictors=['casesNorm','deathsNorm']

tsne.fit(april4th[predictors])

embedding = tsne.embedding_
plt.figure(figsize=(14,7))

plt.subplot(1,2,1)

i=0

for index,row in april4th.iterrows():

    continent=row['continentexp']

    if continent=='Asia':

        plt.scatter(embedding[i,0], embedding[i,1], s=3.0, color='red')

    if continent=='America':

        plt.scatter(embedding[i,0], embedding[i,1], s=3.0, color='blue')

    if continent=='Europe':

        plt.scatter(embedding[i,0], embedding[i,1], s=3.0, color='green')

    if continent=='Africa':

        plt.scatter(embedding[i,0], embedding[i,1], s=3.0, color='cyan')

    if continent=='Oceania':

        plt.scatter(embedding[i,0], embedding[i,1], s=3.0, color='black')

    i=i+1

legend_elements = [Line2D([0], [0], color='w', marker='o', markerfacecolor='red',label='Asia'),

                   Line2D([0], [0], color='w', marker='o', markerfacecolor='blue',label='America'),

                  Line2D([0], [0], color='w', marker='o', markerfacecolor='green',label='Europe'),

                  Line2D([0], [0], color='w', marker='o', markerfacecolor='black',label='Oceania'),

                  Line2D([0], [0], color='w', marker='o', markerfacecolor='cyan',label='Africa')]

plt.legend(handles=legend_elements)



plt.subplot(1,2,2)



i=0

for index,row in april4th.iterrows():

    #plt.annotate(s=row['countriesandterritories'],xy=(embedding[i,0],[i,1]))

    continent=row['continentexp']

    if continent=='Asia':

        plt.scatter(embedding[i,0], embedding[i,1], s=3.0, color='red')

    if continent=='America':

        plt.scatter(embedding[i,0], embedding[i,1], s=3.0, color='blue')

    if continent=='Europe':

        plt.scatter(embedding[i,0], embedding[i,1], s=3.0, color='green')

    if continent=='Africa':

        plt.scatter(embedding[i,0], embedding[i,1], s=3.0, color='cyan')

    if continent=='Oceania':

        plt.scatter(embedding[i,0], embedding[i,1], s=3.0, color='black')

    labels=['Colombia','Peru','Ecuador','Venezuela','Canada','Germany','Singapore','Brazil','China','France']

    for label in labels:

        if row['countriesandterritories']==label:

            plt.annotate(s=label,xy=(embedding[i,0],embedding[i,1])) 

    i=i+1

legend_elements = [Line2D([0], [0], color='w', marker='o', markerfacecolor='red',label='Asia'),

                   Line2D([0], [0], color='w', marker='o', markerfacecolor='blue',label='America'),

                  Line2D([0], [0], color='w', marker='o', markerfacecolor='green',label='Europe'),

                  Line2D([0], [0], color='w', marker='o', markerfacecolor='black',label='Oceania'),

                  Line2D([0], [0], color='w', marker='o', markerfacecolor='cyan',label='Africa')]

plt.legend(handles=legend_elements)
time_evol=pd.DataFrame(index=list(eddc.countriesandterritories.unique()))

countries=eddc.countriesandterritories.unique()

dates=eddc.drop_duplicates(['month','day'])

dates=np.vstack(((np.array(dates['day'])),(np.array(dates['month']))))

dates=dates.T

dates=dates[:45]

nan_col=np.empty(len(countries))

nan_col[:]=np.NaN

time_evol['Continent']=nan_col



for date in dates:

        

    #date=eddc[(eddc['day']==day) & (eddc['month']==month)]

    #if eddc_it['countriesandterritories']==country:

    day=date[0]

    month=date[1]

    #print('Month: ',month,'Day: ',day)

    labelCases='casesNorm '+str(day)+'-'+str(month)

    labelDeaths='deathsNorm '+str(day)+'-'+str(month)

    time_evol[labelCases]=nan_col

    time_evol[labelDeaths]=nan_col

    time_evol['Cases']=nan_col

    time_evol['Deaths']=nan_col

    cases=0

    deaths=0

    temp =eddc[(eddc['day']==day) & (eddc['month']==month)]



    for country,row in temp.iterrows():

        casesNorm=row['casesNorm']

        deathsNorm=row['deathsNorm']

        continent=row['continentexp']

        time_evol.loc[country,labelCases]=casesNorm

        cases+=casesNorm

        time_evol.loc[country,labelDeaths]=deathsNorm

        deaths+=deathsNorm

        time_evol.loc[country,'Continent']=continent

        

        time_evol.loc[country,'Cases']=cases

        time_evol.loc[country,'Deaths']=deaths

#time_evol=time_evol.drop(0,axis='columns')

time_evol=time_evol.dropna(axis=0)

time_evol
tsne = sklearn.manifold.TSNE(perplexity=100)

tsne.fit(time_evol.drop(['Continent','Cases','Deaths'],axis='columns'))

tsne_embedding = tsne.embedding_



plt.figure(figsize=(10,7))

i=0

for index,row in time_evol.iterrows():

    #plt.annotate(s=row['countriesandterritories'],xy=(tsne_embedding[i,0],[i,1]))

    continent=row['Continent']

    if continent=='Asia':

        plt.scatter(tsne_embedding[i,0], tsne_embedding[i,1], s=3.0, color='red')

    if continent=='America':

        plt.scatter(tsne_embedding[i,0], tsne_embedding[i,1], s=3.0, color='blue')

    if continent=='Europe':

        plt.scatter(tsne_embedding[i,0], tsne_embedding[i,1], s=3.0, color='green')

    if continent=='Africa':

        plt.scatter(tsne_embedding[i,0], tsne_embedding[i,1], s=3.0, color='cyan')

    if continent=='Oceania':

        plt.scatter(tsne_embedding[i,0], tsne_embedding[i,1], s=3.0, color='black')

    labels=['Colombia','Peru','Ecuador','Venezuela','Canada','Germany','Singapore','France','United_States_of_America']

    for label in labels:

        if index==label:

            plt.annotate(s=label,xy=(tsne_embedding[i,0],tsne_embedding[i,1])) 

    i=i+1

legend_elements = [Line2D([0], [0], color='w', marker='o', markerfacecolor='red',label='Asia'),

                   Line2D([0], [0], color='w', marker='o', markerfacecolor='blue',label='America'),

                  Line2D([0], [0], color='w', marker='o', markerfacecolor='green',label='Europe'),

                  Line2D([0], [0], color='w', marker='o', markerfacecolor='black',label='Oceania'),

                  Line2D([0], [0], color='w', marker='o', markerfacecolor='cyan',label='Africa')]

plt.legend(handles=legend_elements)

plt.title('t-SNE clustering',size=20)
reducer = umap.UMAP(n_neighbors=5)

reducer.fit(time_evol.drop(['Continent','Deaths','Cases'],axis='columns'))

umap_embedding = reducer.transform(time_evol.drop(['Continent','Deaths','Cases'],axis='columns'))

plt.figure(figsize=(10,7))

i=0

for index,row in time_evol.iterrows():

    #plt.annotate(s=row['countriesandterritories'],xy=(umap_embedding[i,0],[i,1]))

    continent=row['Continent']

    if continent=='Asia':

        plt.scatter(umap_embedding[i,0], umap_embedding[i,1], s=3.0, color='red')

    if continent=='America':

        plt.scatter(umap_embedding[i,0], umap_embedding[i,1], s=3.0, color='blue')

    if continent=='Europe':

        plt.scatter(umap_embedding[i,0], umap_embedding[i,1], s=3.0, color='green')

    if continent=='Africa':

        plt.scatter(umap_embedding[i,0], umap_embedding[i,1], s=3.0, color='cyan')

    if continent=='Oceania':

        plt.scatter(umap_embedding[i,0], umap_embedding[i,1], s=3.0, color='black')

    labels=['Colombia','Peru','Ecuador','Venezuela','Canada','Germany','Singapore','United_States_of_America','Chile','China']

    for label in labels:

        if index==label:

            plt.annotate(s=label,xy=(umap_embedding[i,0],umap_embedding[i,1])) 

    i=i+1

legend_elements = [Line2D([0], [0], color='w', marker='o', markerfacecolor='red',label='Asia'),

                   Line2D([0], [0], color='w', marker='o', markerfacecolor='blue',label='America'),

                  Line2D([0], [0], color='w', marker='o', markerfacecolor='green',label='Europe'),

                  Line2D([0], [0], color='w', marker='o', markerfacecolor='black',label='Oceania'),

                  Line2D([0], [0], color='w', marker='o', markerfacecolor='cyan',label='Africa')]

plt.legend(handles=legend_elements)

plt.title('UMAP clustering',size=20)
net = torch.nn.Sequential(

                torch.nn.Linear(90, 40),

                torch.nn.ReLU(),

                torch.nn.Linear(40, 20),

                torch.nn.ReLU(),

                torch.nn.Linear(20, 10),

                torch.nn.ReLU(),

                torch.nn.Linear(10, 5),

                torch.nn.ReLU(),

                torch.nn.Linear(5, 1)

)

#Dado que no es un problema de clasificación, debemos utilizar un criterio diferente. MSELoss mide la distancia entre la

#predicción y el valor verdadero

criterion = torch.nn.MSELoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.02) #lr: learning rate
epochs = 100



excep_deaths=[s for s in time_evol.keys() if 'deaths' in s]

only_cases=time_evol.drop(excep_deaths,axis='columns')

target_cases=only_cases['Cases']

only_cases=time_evol.drop(['Continent','Cases','Deaths'],axis='columns')

loss_values = np.zeros((len(only_cases),epochs))



it=0

only_cases_countries=[]

for index,row in only_cases.iterrows():

    inputs = torch.autograd.Variable(torch.Tensor(row.values).float())

    targets = torch.autograd.Variable(torch.Tensor(row.values).float())

    for epoch in range(epochs):



        optimizer.zero_grad()

        out = net(inputs)

        loss = criterion(out, targets)

        loss.backward()

        optimizer.step()



        loss_values[it,epoch] = loss.item()

    only_cases_countries.append(index)

    it+=1
plt.figure(figsize=(20,20))

plt.imshow(loss_values.T)

plt.colorbar()

plt.xlabel('Countries',size=20)

plt.ylabel('Epochs',size=20)

plt.title('Loss criterion for training country by country for 100 epochs',size=24)

plt.xticks(np.arange(len(only_cases_countries)),only_cases_countries,rotation=90,size=9)

plt.show()
casesCol =time_evol.loc['Colombia',[s for s in time_evol.keys() if 'cases' in s]]

casesCol= casesCol[::-1]

i=1

while i<45:

    casesCol[i]=casesCol[i]+casesCol[i-1]

    i+=1

deathsCol =time_evol.loc['Colombia',[s for s in time_evol.keys() if 'deaths' in s]]

deathsCol= deathsCol[::-1]

i=1

while i<45:

    deathsCol[i]=deathsCol[i]+deathsCol[i-1]

    i+=1



strDates=[]

for date in dates:

    strDates.append(str(date[0])+'/'+str(date[1]))

strDates= strDates[::-1]



plt.figure(figsize=(14,10))

UncertCol=np.array(casesCol+12*(loss_values[99,only_cases_countries.index('Colombia')])**0.5)

DowncertCol=np.array(casesCol-12*(loss_values[99,only_cases_countries.index('Colombia')])**0.5)



dias=np.array(np.arange(len(strDates)))

plt.plot(dias,casesCol,'.')

#plt.plot(dias,UncertCol,'.')

plt.fill_between(x=dias,y1=UncertCol.astype(float),y2=DowncertCol.astype(float),alpha=0.2)

plt.xticks(dias,strDates,rotation=45)

plt.xlabel('Date',size=20)

plt.ylabel('Covid cases per capita',size=20)

plt.title('Confiedence interval for Covid cases in Colombia',size=24)

plt.show()