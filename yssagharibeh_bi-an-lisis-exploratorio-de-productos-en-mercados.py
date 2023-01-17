!pip install plotly==4.5.2
#Helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Viz

import seaborn as sns #Viz



import plotly.graph_objs as go

import plotly.express as px #Viz



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/productos-consumo-masivo/output - Kaggle.xlsx', decimal=',')

# Print the head of df

df.head()
# Create the top_10_nation pandas series

by_subcategory = df.subcategory.value_counts()

top_10_subcat = by_subcategory[:10]

top_10_subcat
top_10_subcat.plot(kind='barh');
df.groupby('subcategory').prod_unit_price.agg([min, max]).sort_values(by=['min', 'max'], ascending=False)
# Group by category

category_group = df[['prod_brand','prod_unit_price']].groupby('prod_brand').mean()

category_group.sort_values(by=['prod_unit_price'])

## Precio promedio por la Marca
# Print the info of df

print(df.info())



# Print the shape of df

print(df.shape)
# Statistics for continuous variables

df.describe()
# Statistics for categorical variables

print(pd.DataFrame(df['date'].value_counts(dropna=False)))

df.describe(include=[np.object])
# Distribution by date and supermarket

data = df.groupby(['category', 'subcategory']).size()

sns.barplot(data.values, data.index, palette="Blues")
# Different category distributions by size

data = pd.DataFrame({'count' : df[df['date'] == 20190709].groupby(['subcategory', 'tags']).size()}).reset_index()

fig = px.treemap(data, path=['subcategory', 'tags'], values='count')

fig.show()



 
# Different category distributions by mean price

data = pd.DataFrame({'value' : df[df['date'] == 20190709].groupby(['subcategory']).prod_unit_price.mean()}).reset_index()

data = data.sort_values(['value'],ascending=False).reset_index(drop=True)

sns.barplot(data['value'], data['subcategory'], palette="Blues")
# Different tag distributions by mean price. More expensive

plt.figure(figsize = (5, 10))

data = pd.DataFrame({'value' : df[df['date'] == 20190709].groupby(['tags']).prod_unit_price.mean()}).reset_index()

data = data.sort_values(['value'],ascending=False).reset_index(drop=True).head(30)

sns.barplot(data['value'], data['tags'], palette="Blues")
# Different tag distributions by mean price. Less expensive

plt.figure(figsize = (5, 10))

data = pd.DataFrame({'value' : df[df['date'] == 20190709].groupby(['tags']).prod_unit_price.mean()}).reset_index()

data = data.sort_values(['value'],ascending=False).reset_index(drop=True).tail(30)

sns.barplot(data['value'], data['tags'], palette="Blues")
# Different category distributions by standard deviation

data = pd.DataFrame({'std' : df[df['date'] == 20190709].groupby(['subcategory', 'tags']).prod_unit_price.std()}).reset_index()

fig = px.treemap(data, path=['subcategory', 'tags'], values='std')

fig.show()
# Different category distributions by standard deviation

data = df[df['date'] == 20190709]

sns.boxplot(x="prod_unit_price", y="subcategory", data=data)
# Different tags distributions without Vinos y Licores

data = pd.DataFrame({'mean' : df[(df['date'] == 20190709) & (df['subcategory'] != 'Lácteos, huevos y refrigerados') & (df['subcategory'] != 'Productos Congelados')].groupby(['tags']).prod_unit_price.mean()}).reset_index()

data = data.sort_values(['mean'],ascending=False).reset_index(drop=True)

order = data['tags']



data = df[(df['date'] == 20190709) & (df['subcategory'] != 'Lácteos, huevos y refrigerados') & (df['subcategory'] != 'Productos Congelados')]

plt.figure(figsize = (10, 30))



sns.boxplot(x="prod_unit_price", y="tags", data=data, order= order)
# Data standardization with mean

data = df[(df['date'] == 20190709) & (df['subcategory'] != 'Lácteos, huevos y refrigerados') & (df['subcategory'] != 'Productos Congelados')]

means = pd.DataFrame({'mean' : df[(df['date'] == 20190709) & (df['subcategory'] != 'Lácteos, huevos y refrigerados') & (df['subcategory'] != 'Productos Congelados')].groupby(['tags']).prod_unit_price.mean()}).reset_index()



data = data.merge(means, on='tags')

data['std_price'] = data['prod_unit_price']/data['mean']

data.head(3)
# Different tags distributions without Vinos y Licores after standardization

order = pd.DataFrame({'std' : data.groupby(['tags']).std_price.std()}).reset_index()

order = order.sort_values(['std'],ascending=False).reset_index(drop=True)

order = order['tags']



plt.figure(figsize = (10, 30))

sns.boxplot(x="std_price", y="tags", data=data, order= order)
# Inspection of water

df[(df['date'] == 20190709) & (df['tags'] == 'Agua')]
# Data standardization with mean

data = df[(df['prod_source'] == 'VERDE') & (df['subcategory'] != 'Lácteos, huevos y refrigerados') & (df['subcategory'] != 'Productos Congelados')]

means = pd.DataFrame({'mean' : df[(df['prod_source'] == 'AMARILLO') & (df['subcategory'] != 'Lácteos, huevos y refrigerados') & (df['subcategory'] != 'Productos Congelados')].groupby(['prod_name']).prod_unit_price.mean()}).reset_index()



data = data.merge(means, on='prod_name')

data['std_price'] = data['prod_unit_price']/data['mean']

data.head(50)