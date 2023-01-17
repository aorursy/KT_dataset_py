#importing libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans
%matplotlib inline

import warnings 
warnings.filterwarnings('ignore')
#getting data

data_2017 = pd.read_csv('../input/ranking_2017.csv', sep=',')

data_2017.set_index('Ranking', inplace=True)

data_2017.head()
#filtering

grades = data_2017[['Nome','Pública ou Privada','Estado','Nota em Ensino','Nota em Pesquisa',
                    'Nota em Mercado','Nota em Inovação','Nota em Internacionalização','Nota']]

grades.head(10)
#correlations

plt.figure(figsize=(10, 10))
crr = grades.corr()
sns.heatmap(crr, annot=True, linewidth=.5)
#research vs teaching

sns.lmplot(x='Nota em Ensino', y='Nota em Pesquisa', fit_reg=True, data=grades, size=5, aspect=2)
#industry income vs teaching

sns.lmplot(x='Nota em Ensino', y='Nota em Mercado', data=grades, aspect=2)
#innovation vs research

sns.lmplot(x='Nota em Pesquisa', y='Nota em Inovação', data=grades, aspect=2)
#international outlook vs research

sns.lmplot(x='Nota em Pesquisa', y='Nota em Internacionalização', data=grades,aspect=2)
#filtering & preparing data

def converter(change):
    if change == 'Privada':
        return 'Privado'
    else:
        return 'Público'
    
industry_income = data_2017[['Nome','Pública ou Privada','Nota em Mercado', 'Nota']]

industry_income['Pública ou Privada'] = industry_income['Pública ou Privada'].apply(converter)

industry_income.head(15)
#distribution

plt.figure(figsize=(14, 5))

sns.distplot(industry_income['Nota em Mercado'].dropna(), bins=30, kde=True, color="b")
#importing libraries

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.graph_objs as go
init_notebook_mode(connected=True)

cf.go_offline()
#preparing data frame
spread = industry_income.head(55)
spread.reset_index(inplace=True)

# Creating trace1
trace1 = go.Scatter(
                    x = spread['Ranking'],
                    y = spread['Nota em Mercado'],
                    mode = "lines",
                    name = "Industry Income",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                    text= spread['Nome'])
# Creating trace2
trace2 = go.Scatter(
                    x = spread['Ranking'],
                    y = spread['Nota'],
                    mode = "lines",
                    name = "Final Score",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= spread['Nome'])
data = [trace1, trace2]
layout = dict(title = 'Industry Income and Final Score vs Ranking',
              xaxis= dict(title= 'Ranking Universitário Folha',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
#industry income vs final score

sns.lmplot(x='Nota', y='Nota em Mercado', data=industry_income, aspect=2, hue='Pública ou Privada', fit_reg=True)
#preparing data frame
a = industry_income[industry_income['Pública ou Privada']=='Público']
b = industry_income[industry_income['Pública ou Privada']=='Privado']

# Creating trace1
trace01 = go.Scatter(
    x = a['Nota em Mercado'],
    y = a['Nota'],
    name = 'Público',
    mode = 'markers',
    marker = dict(
        size = 8,
        color = 'rgba(16, 112, 2, 0.8)',
        line = dict(
            width = 1,
            
        )
    )
)

# Creating trace2
trace02 = go.Scatter(
    x = b['Nota em Mercado'],
    y = b['Nota'],
    name = 'Privado',
    mode = 'markers',
    marker = dict(
        size = 8,
        color = 'rgba(80, 26, 80, 0.8)',
        line = dict(
            width = 1,
        )
    )
)

data = [trace01, trace02]

layout = dict(title = 'Public or Private in Final Score vs Industry Income',
              yaxis = dict(zeroline = False),
              xaxis = dict(title= 'Ranking Universitário Folha', zeroline = False)
             )

fig = dict(data=data, layout=layout)
iplot(fig, filename='styled-scatter')
#preparing data and kmeans application

features = np.array(data_2017[['Nota em Ensino','Nota em Pesquisa', 'Nota em Mercado',
                               'Nota em Inovação','Nota em Internacionalização']].fillna(value=0))

kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit(features)

#organizing data

grades['Agrupamento'] = kmeans.labels_

def converter1(change):
    if change == 1:
        return 'Grupo 1'
    else:
        return 'Grupo 2'

grades['Agrupamento'] = grades['Agrupamento'].apply(converter1)

sns.lmplot(x='Nota', y='Nota em Mercado', data=grades, aspect=2, hue='Agrupamento', fit_reg=False)
#preparing data and kmeans application

kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(features)

#organizing data

grades['Agrupamento'] = kmeans.labels_

def converter2(change):
    if change == 1:
        return 'Grupo 3'
    elif change == 2:
        return 'Grupo 1'
    else:
        return 'Grupo 2'

grades['Agrupamento'] = grades['Agrupamento'].apply(converter2)

sns.lmplot(x='Nota', y='Nota em Mercado', data=grades, aspect=2, hue='Agrupamento', fit_reg=False)
