!pip install plotly==4.6.0

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt #Viz

import seaborn as sns #Viz

import plotly.express as px #Viz



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
base = pd.read_excel('/kaggle/input/productos-consumo-masivo/output - Kaggle.xlsx', decimal=',')

base.head(5)
# Analizando la estructura del archivo suministrado:

print(base.info())



# Print the shape of df

print(base.shape)
base.describe()
# Statistics for categorical variables

print(pd.DataFrame(base['date'].value_counts(dropna=False)))

base.describe(include=[np.object])
# Distribution by date and supermarket

data = base.groupby(['date', 'prod_source']).size()

sns.barplot(data.values, data.index, palette="Blues")
#Distribution by product in a specific date

data = base[base['date'] == 20190609].groupby(['subcategory']).size()

sns.barplot(data.values, data.index, palette="Blues")
#Distribution by product in a specific date

data = base[base['date'] == 20190709].groupby(['subcategory']).size()

sns.barplot(data.values, data.index, palette="Blues")
#Distribution by product in a specific date

data = base[base['date'] == 20191101].groupby(['subcategory']).size()

sns.barplot(data.values, data.index, palette="Blues")
#Distribution by product in a specific date

data = base[base['date'] == 20200220].groupby(['subcategory']).size()

sns.barplot(data.values, data.index, palette="Blues")
# Different category distributions by size

data = pd.DataFrame({'count' : base[base['date'] == 20190709].groupby(['subcategory', 'tags']).size()}).reset_index()

fig = px.treemap(data, path=['subcategory', 'tags'], values='count')

fig.show()
data = base[base['date'] == 20190609]

sns.boxplot(x="prod_unit_price", y="subcategory", data=data)
data = base[base['date'] == 20190709]

sns.boxplot(x="prod_unit_price", y="subcategory", data=data)
data = base[base['date'] == 20191101]

sns.boxplot(x="prod_unit_price", y="subcategory", data=data)
data = base[base['date'] == 20200220]

sns.boxplot(x="prod_unit_price", y="subcategory", data=data)
# Different tags distributions without Vinos y Licores

data = pd.DataFrame({'mean' : base[(base['date'] == 20190709) & (base['subcategory'] != 'Vinos y Licores') & (base['subcategory'] != 'Charcutería')].groupby(['tags']).prod_unit_price.mean()}).reset_index()

data = data.sort_values(['mean'],ascending=False).reset_index(drop=True)

order = data['tags']



data = base[(base['date'] == 20190709) & (base['subcategory'] != 'Vinos y Licores') & (base['subcategory'] != 'Charcutería')]

plt.figure(figsize = (10, 30))

sns.boxplot(x="prod_unit_price", y="tags", data=data, order= order)

