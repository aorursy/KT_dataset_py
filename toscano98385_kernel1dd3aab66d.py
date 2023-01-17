# Importacion de librerias y de visualizacion (matplotlib y seaborn)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt



%matplotlib inline



plt.style.use('default') # para graficos matplotlib

plt.rcParams['figure.figsize'] = (10, 8)



sns.set(style="whitegrid") # grid seaborn



pd.options.display.float_format = '{:20,.3f}'.format # notacion output
path = "/home/seba/Escritorio/Datos/TP1/data/"

df_props_full = pd.read_csv(path + "train_dollar.csv")
df_props_full.columns
df_props_full['fecha'] = pd.to_datetime(df_props_full['fecha'])
# Convierto todos los valores 1/0 a uint8

df_props_full['gimnasio'] = df_props_full['gimnasio'].astype('uint8')

df_props_full['usosmultiples'] = df_props_full['usosmultiples'].astype('uint8')

df_props_full['piscina'] = df_props_full['piscina'].astype('uint8')

df_props_full['escuelascercanas'] = df_props_full['escuelascercanas'].astype('uint8')

df_props_full['centroscomercialescercanos'] = df_props_full['centroscomercialescercanos'].astype('uint8')
# Convierto los representables en uint8. Utilizo el tipo de pandas UInt8Dtype para evitar conflicto con NaN

df_props_full['antiguedad'] = df_props_full['antiguedad'].astype(pd.UInt8Dtype())

df_props_full['habitaciones'] = df_props_full['habitaciones'].astype(pd.UInt8Dtype())

df_props_full['garages'] = df_props_full['garages'].astype(pd.UInt8Dtype())

df_props_full['banos'] = df_props_full['banos'].astype(pd.UInt8Dtype())
# Convierto los representables en uint16. Utilizo el tipo de pandas UInt16Dtype para evitar conflicto con NaN

df_props_full['metroscubiertos'] = df_props_full['metroscubiertos'].astype(pd.UInt16Dtype())

df_props_full['metrostotales'] = df_props_full['metrostotales'].astype(pd.UInt16Dtype())
# Convierto los representables en uint32. Utilizo el tipo de pandas UInt32Dtype para evitar conflicto con NaN

df_props_full['id'] = df_props_full['id'].astype(pd.UInt32Dtype())

df_props_full['idzona'] = df_props_full['idzona'].astype(pd.UInt32Dtype())

df_props_full['Precio_MEX'] = df_props_full['Precio_MEX'].astype(pd.UInt32Dtype())

df_props_full['Precio_USD'] = df_props_full['Precio_USD'].astype(pd.UInt32Dtype())
df_props_full['year'] = df_props_full['fecha'].dt.year

df_props_full['month'] = df_props_full['fecha'].dt.month
df_props_full.loc[df_props_full['year']==2016].loc[df_props_full['month']==12].shape
df_props_full['first_fortnight'] = df_props_full['fecha'].apply(lambda fecha: 1 if fecha.day < 15 else 0)
df_props_full.groupby('first_fortnight').agg('size').to_frame()
# Convierto fecha en un numero siendo 2012-01 = 1, 2012-02 = 2 ... 2013-01 = 13

def name_date(date):

    year_from_2012 = (date.year - 2012) * 12

    month = date.month

    return year_from_2012 + month
periods = df_props_full['fecha'].apply(name_date)

periods = periods.to_frame()
periods.groupby('fecha').size().to_frame()
periods.hist(bins=60)

plt.title("Crecimiento ZonaProp entre 2012 y 2016", fontsize = 15)

plt.xticks(ticks=np.arange(0,65,10))

plt.xlabel("Mes", fontsize = 12)

plt.ylabel("Cantidad de Publicaciones", fontsize = 12)
df_reduced = df_props_full.loc[:,{'year','month'}]

df_reduced.head(2)
df_reduced = df_reduced.groupby(['year','month']).agg('size').to_frame()

df_reduced = df_reduced.reset_index()
pubs_by_month_and_year = pd.pivot_table(df_reduced, index = ['month'], columns = ['year'])

pubs_by_month_and_year
pubs_by_month_and_year.columns = pubs_by_month_and_year.columns.droplevel()
pubs_by_month_and_year
pubs_by_month_and_year.plot(kind='area',legend=True)

plt.legend(bbox_to_anchor = (1.2, 1))

plt.title("Publicaciones a lo largo del tiempo", fontsize = 15)

# plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12])

plt.xlabel("Mes", fontsize = 12)

plt.ylabel("Cantidad de Publicaciones", fontsize = 12)
df_transposed = pubs_by_month_and_year.transpose()
df_transposed.columns.name = 'Mes'
fig, axes = plt.subplots(2,2, sharex='all', sharey='row')

df_transposed.iloc[0].plot.bar(ax=axes[0,0], color='red')

df_transposed.iloc[1].plot.bar(ax=axes[0,1], color='orange')

df_transposed.iloc[2].plot.bar(ax=axes[1,0], color='purple')

df_transposed.iloc[3].plot.bar(ax=axes[1,1], color='blue')

# df_transposed.iloc[4].plot.bar(ax=axes[2,0], color='green')



axes[0,0].set_title('2012')

axes[0,1].set_title('2013')

axes[1,0].set_title('2014')

axes[1,1].set_title('2015')



fig.suptitle('Publicaciones por mes', fontsize=16)

plt.show()
df_transposed.iloc[len(df_transposed)-1].to_frame().plot.bar(color='green', legend=False)

plt.title("Publicaciones en 2016", fontsize = 15)
