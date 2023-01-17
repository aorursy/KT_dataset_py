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

casa = pd.read_csv('/kaggle/input/housedata/data.csv')



casa ['bedrooms'] = casa ['bedrooms'].astype(int)

casa ['bathrooms'] = casa ['bathrooms'].astype(int)

casa ['floors'] = casa ['floors'].astype(int)

casa ['city'] = casa ['city'].astype('category')







casa.head()
casa.describe()
casa['price'].min() 

casa['price'].max()
casa['bathrooms'].max()
casa['waterfront'].sum()
casa['city'].nunique()
casas=casa ['city'].mode()

print('cidade com mais casas',casas[0])
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

plt.figure (figsize=(20,20))

sns.countplot(data=casa, y='city')

plt.show()

plt.show()
corr = casa[['bedrooms','price']]

corr.corr()
print(casa['floors'].value_counts())

plt.pie((casa['floors'].value_counts()).sort_values(), labels=['3','2','1'])

plt.show()
casa_no = casa[(casa['bedrooms']==3) & (casa['city']=='Seattle')]

casa_no.head()

print(casa_no.std()) #desvio padrão

print(casa_no.mean()) #media
import plotly.offline as py

import plotly.graph_objs as go

trace = go.Scatter(x = casa['bathrooms'],

                   y = casa['price'],

                   mode = 'markers')

data = [trace]

py.iplot(data)

layout = go.Layout(title='Numero de banheiros x Preço de venda',

                   yaxis={'title':'Preço da casa'},

                   xaxis={'title': 'Banheiros'})

# Criando figura que será exibida

fig = go.Figure(data=data, layout=layout)

# Exibindo figura/gráfico

py.iplot(fig)
corr = casa[['bedrooms','price']]

corr.corr()
banheiro = (casa['bathrooms'].value_counts()).sort_values()

print(banheiro)

print(banheiro[1])

print(banheiro[2])

print(banheiro[3])

print(banheiro[4])

print(banheiro[5])



taxa_banheiro = banheiro[1]/(banheiro[0]+banheiro[1])*100

texto1= "numeros de banheiros por casas  = "+ str(round(taxa_banheiro,2)) + "%"



#plt.bar(x, height, width=0.8, color='red')

plt.bar(8,banheiro[8], width = 0.9,  color='blue')

plt.bar(6,banheiro[6], width = 0.9,  color='red')

plt.bar(5,banheiro[5], width = 0.9,  color='yellow')

plt.bar(0,banheiro[0], width = 0.9,  color='blue')

plt.bar(4,banheiro[4], width = 0.9,  color='red')

plt.bar(3,banheiro[3], width = 0.9,  color='yellow')

plt.bar(1,banheiro[1], width = 0.9,  color='blue')

plt.bar(2,banheiro[2], width = 0.9,  color='red')



plt.title('número de banheiros por casas ')



plt.text(0.0,1700,texto1)

plt.xticks([8,6,5,0,4,3,1,2],['8','6','5','0','4','3','1','2'])

plt.xticks()

plt.grid(True)

plt.show()