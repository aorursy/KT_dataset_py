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
precio_antiguedad = df_props_full.groupby('antiguedad').agg({'Precio_USD':'mean'})

precio_antiguedad = precio_antiguedad.rename(columns={'Precio_USD':'Precio_promedio_USD'})

precio_antiguedad = precio_antiguedad.reset_index()



precio_fig = sns.barplot(data=precio_antiguedad, x='antiguedad',y='Precio_promedio_USD', orient='v', palette = (sns.color_palette("viridis",)))

precio_fig.set_title("Precio promedio según Antigüedad", fontsize = 15)

precio_fig.set_ylabel("Precio Promedio (USD)", fontsize = 12)

precio_fig.set_xlabel("Años Antigüedad Propiedad", fontsize = 12)



for ind, label in enumerate(precio_fig.get_xticklabels()):

    if ind %  10 == 0:  # every 10th label is kept

        label.set_visible(True)

    else:

        label.set_visible(False)
df_props_full['year'] = df_props_full['fecha'].dt.year

df_props_full['month'] = df_props_full['fecha'].dt.month
precios_x_mes_mex = df_props_full.groupby(['year','month']).agg({'Precio_MEX':'mean'}).reset_index()

precios_x_mes_mex = pd.pivot_table(precios_x_mes_mex, index='year', columns=['month'])

precios_x_mes_mex.columns = precios_x_mes_mex.columns.droplevel()

precios_x_mes_mex
# Convierto fecha en un numero siendo 2012-01 = 1, 2012-02 = 2 ... 2013-01 = 13

def monthFrom2012_2016(date):

    year_from_2012 = (date.year - 2012) * 12

    month = date.month

    return year_from_2012 + month
historial_precio_mex = df_props_full.loc[:,['fecha','Precio_MEX']]

historial_precio_mex['fecha'] = historial_precio_mex['fecha'].apply(monthFrom2012_2016)

historial_precio_mex = historial_precio_mex.groupby('fecha').agg({'Precio_MEX':'mean'}).reset_index()

historial_precio_mex.columns = ['Mes','Precio_MEX_Promedio']

historial_precio_mex = historial_precio_mex.set_index('Mes')

historial_precio_mex
historial_precio_mex.plot.line(legend=False, color='green', linewidth=3)

plt.title("Evolución Precio Promedio (MXN) entre 2012 y 2016", fontsize = 15)

plt.xticks(ticks=np.arange(0,65,5))

plt.xlabel("Mes", fontsize = 12)

plt.ylabel("Precio Promedio (MXN)", fontsize = 12)
precios_x_mes_usd = df_props_full.groupby(['year','month']).agg({'Precio_USD':'mean'}).reset_index()

precios_x_mes_usd = pd.pivot_table(precios_x_mes_usd, index='year', columns=['month'])

precios_x_mes_usd.columns = precios_x_mes_usd.columns.droplevel()

precios_x_mes_usd
historial_precio_usd = df_props_full.loc[:,['fecha','Precio_USD']]

historial_precio_usd['fecha'] = historial_precio_usd['fecha'].apply(monthFrom2012_2016)

historial_precio_usd = historial_precio_usd.groupby('fecha').agg({'Precio_USD':'mean'}).reset_index()

historial_precio_usd.columns = ['Mes','Precio_USD_Promedio']

historial_precio_usd = historial_precio_usd.set_index('Mes')
historial_precio_usd.plot.line(legend=False, color='c', linewidth=3)

plt.title("Evolución Precio Promedio (USD) entre 2012 y 2016", fontsize = 15)

plt.xticks(ticks=np.arange(0,65,5))

plt.xlabel("Mes", fontsize = 12)

plt.ylabel("Precio Promedio (USD)", fontsize = 12)
df_dollar = pd.read_csv(path + 'dollar.csv')

df_dollar = df_dollar.dropna()

df_dollar['Cierre'] = pd.to_numeric(df_dollar['Cierre'])

df_dollar['Cierre'] = df_dollar['Cierre'].round(3)

df_dollar['Fecha'] = pd.to_datetime(df_dollar['Fecha'], format='%d.%m.%Y')

df_dollar = df_dollar.set_index('Fecha')

df_dollar = df_dollar.loc[:, 'Cierre'].to_frame()
# Agrego fechas faltantes (Sabados y Domingos) con valor 0

idx = pd.date_range(start='2011-12-12', end='2017-01-31')

df_dollar = df_dollar.reindex(idx, fill_value=0)
# Cuando se trata de una fecha que corresponde a un Sabado o Domingo no se tiene infromación sobre Cierre

# Le asigno el valor correspondiente al Viernes previo

for i in range(0, len(df_dollar)):

    if (df_dollar.iloc[i]['Cierre'] == 0):

        df_dollar.iloc[i]['Cierre'] = df_dollar.iloc[i-1]['Cierre']
df_dollar['Cierre'] = 1 / df_dollar['Cierre']

df_dollar = df_dollar.loc['2012-01-01':'2016-12-31'] 

df_dollar.head(2)
df_dollar = df_dollar.reset_index()

df_dollar.head(1)
df_dollar = df_dollar.rename(columns={'index':'fecha'})

df_dollar.head(1)
df_dollar['fecha'] = pd.to_datetime(df_dollar['fecha'], format='%Y.%m.%d')

df_dollar['fecha'] = df_dollar['fecha'].apply(monthFrom2012_2016)
df_dollar.head(40)
dollarEvolution = df_dollar.groupby('fecha').agg({'Cierre':'mean'})

dollarEvolution
dollarEvolution.plot.line(legend=False, color='r', linewidth=3)

plt.title("Evolución Relación Dólar-Peso Mexicano entre 2012 y 2016", fontsize = 15)

plt.xticks(ticks=np.arange(0,65,5))

plt.yticks(ticks=np.arange(12,22,1))

plt.xlabel("Mes", fontsize = 12)

plt.ylabel("Precio del Dólar (MXN)", fontsize = 12)