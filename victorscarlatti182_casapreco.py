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
dados = pd.read_csv('/kaggle/input/housedata/data.csv')       

        
dados['bedrooms'] = dados['bedrooms'].astype(int)

dados['bathrooms'] = dados['bathrooms'].astype(int)

dados['floors'] = dados['floors'].astype(int)

dados['city'] = dados['city'].astype('category')
dados.describe()





minimo = dados['price'].min()

maximo = dados['price'].max()



print("Valor Minimo:", minimo)



print("Valor Maximo:", maximo)
qtd_banheiros_max = dados['bathrooms'].max()



print("Maior numero de banheiros em uma casa:", qtd_banheiros_max)
dados['waterfront'].sum()
dados['city'].nunique()







dados['city'].mode()

import matplotlib.pyplot as plt # importando a biblioteca gráfica

plt.hist(dados['price'].value_counts(), bins=60, facecolor='red', alpha=0.5, rwidth=0.8)

plt.title('Histograma de numeros de casas X preço ')

plt.xlabel("preço em k dolares")

plt.ylabel("quantidade de casas")

plt.text(10,1370, "OBS: preços são diferentes demais para se acumularem ")

plt.text(10,1270, "em grafico. Portanto os relevos de preços distintos")

plt.text(10,1170, "uns dos outros é invisivel neste histograma")

plt.xlim(0, 100)

plt.ylim(0, 1500)

plt.show()





dados.corr()
dados['price'].corr(dados['bedrooms'])
dados.columns

yy = dados['floors'].value_counts()

plt.title("Porcentagem de Casas com 1, 2 e 3 quartos")

plt.pie(dados['floors'].value_counts(), autopct="%.1f%%")

plt.show()



dados['floors'].value_counts()
resp = dados[(dados['city'] == 'Seattle') & (dados['bedrooms']==3) ]

resp.head()



media = resp['price'].median()

desvio = resp['price'].std()

print("Essa é a media de preço destas casas:",media)

print("Esse é o desvio padrão destas casas:", desvio)
plt.scatter(dados['bathrooms'], dados['price'])

plt.show()
dados['price'].corr(dados['bathrooms'])
dstr = dados['bathrooms'].value_counts()

plt.bar(0,dstr[0], width=0.8)

plt.bar(1,dstr[1], width=0.8)

plt.bar(2,dstr[2], width=0.8)

plt.bar(3,dstr[3], width=0.8)

plt.bar(4,dstr[4], width=0.8)

plt.bar(5,dstr[5], width=0.8)

plt.bar(6,dstr[6], width=0.8)

plt.bar(7,dstr[8], width=0.8)

plt.title('Quantidade de casas com cada numero de banheiro')

plt.xticks([0,1,2,3,4,5,6,7],['1 ','2','3','4 ','5','6','7','8'])

plt.ylim(2, 2500)

plt.grid(True)

plt.show()