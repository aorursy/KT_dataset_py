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
casa = pd.read_csv('/kaggle/input/housedata/data.csv')            

casa.head(5)
casa.describe()
print("Valor máximo: ", casa['price'].astype(int).max())

print("Valor mínimo: ", casa ['price'].astype(int).min())

print("Maior número de banheiros", casa['bathrooms'].astype(int).max())
casa['waterfront'].count()
casa['city'].astype('category').value_counts()

casa['bedrooms'] = casa['bedrooms'].astype(int)

casa['bathrooms'] = casa['bathrooms'].astype(int)

casa['floors'] = casa['floors'].astype(int)
casa['city'].max()
import matplotlib.pyplot as plt

casas = casa['city']

casas_sum = casas.value_counts()

casas_sum = casas_sum.sort_index()

x = casas_sum.index

y = casas_sum

plt.figure(figsize=(60, 25))

plt.title("Número de casas por cidade")

plt.bar(x,y)

import matplotlib.pyplot as plt

quartos = casa['bedrooms']

precos = casa['price']

plt.scatter(quartos,precos)

plt.title('Número de Quartos X Preço')

plt.xlabel("Quartos")

plt.ylabel("Preços")

plt.show()



import matplotlib.pyplot as plt

plt.pie(casa['floors'].value_counts(), autopct='%1.1f%%', labels=[1,2,3], labeldistance=1.3)

plt.show()
seatle_house = casa[(casa['bathrooms']==3) & (casa['city']=='Seattle')]

seatle_house.head()

print(seatle_house.std())

print(seatle_house.mean())
import plotly.offline as py

import plotly.graph_objs as gb



house = gb.Scatter(x = casa['bathrooms'], y= casa['price'],

                  mode = 'markers')



data=[house]

py.iplot(data)

layout = gb.Layout(title="numero de banheiros x preco de vendas",

                  yaxis="Preco das casas"

                  ,xaxis= "banheiros")



fig = gp.Figure(data=data, layout=layout)

py.iplot(fig)
cont = casa['bathrooms'].value_counts()

cidade = cont.index



plt.figure(figsize=(15,10))



for n, i in enumerate(cont):

    plt.barh(n,i)

    plt.text(i+20, n, cidade[n])

plt.ylabel('banheiro')

plt.xlabel('cidades')



plt.grid()

plt.show()