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



pd.options.display.float_format = '{:20,.0f}'.format # notacion output
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
tipos = df_props_full['tipodepropiedad'].value_counts()

tipos
len(tipos)
type_fig = sns.barplot(x=tipos, y=tipos.index, orient='h', palette = (sns.color_palette("gist_stern", 20)))

type_fig.set_title("Cantidad de publicaciones según Tipo de Propiedad", fontsize = 15)

type_fig.set_xlabel("Cantidad Publicaciones", fontsize = 12)
tipos_inf = tipos[4:-1]

tipos_inf
type_inf_fig = sns.barplot(x=tipos_inf, y=tipos_inf.index, orient='h', palette = (sns.color_palette("gist_stern", 20)))

type_inf_fig.set_title("Tipos de Propiedad con menos Publicaciones", fontsize = 15)

type_inf_fig.set_xlabel("Cantidad Publicaciones", fontsize = 12)
antiguedades = df_props_full['antiguedad'].value_counts()

antiguedades = antiguedades.sort_index()

antiguedades
antig_new = antiguedades[0:39]

antig_old = antiguedades[39:len(antiguedades)]
antig_fig = sns.barplot(x=antig_new.index, y=antig_new, orient='v', palette = (sns.color_palette("gist_stern", 20)))

antig_fig.set_title("Cantidad de Publicaciones según Antigüedad de Propiedades", fontsize = 15)

antig_fig.set_ylabel("Cantidad Publicaciones", fontsize = 12)

antig_fig.set_xlabel("Años Antigüedad", fontsize = 12)
antig_old_fig = sns.barplot(x=antig_old.index, y=antig_old, orient='v', palette = (sns.color_palette("gist_stern", 20)))

antig_old_fig.set_title("Cantidad de Publicaciones según Antigüedad de Propiedades", fontsize = 15)

antig_old_fig.set_ylabel("Cantidad Publicaciones", fontsize = 12)

antig_old_fig.set_xlabel("Años Antigüedad", fontsize = 12)
tiposPorProvincia = df_props_full.groupby(['tipodepropiedad','provincia']).size().reset_index()

tiposPorProvincia
tiposPorProvincia = pd.pivot_table(tiposPorProvincia, index='tipodepropiedad', columns=['provincia'])

tiposPorProvincia
tiposPorProvincia = tiposPorProvincia.fillna(0)

tiposPorProvincia
tiposPorProvincia.columns = tiposPorProvincia.columns.droplevel()
sns.heatmap(tiposPorProvincia.T, cmap='inferno', annot = False)
precioPorTipo = df_props_full.loc[:,['tipodepropiedad','Precio_USD']]

precioPorTipo
precioPromedio = precioPorTipo.groupby('tipodepropiedad').agg({'Precio_USD':'mean'}).reset_index().sort_values('Precio_USD',ascending=False)
precioPromedio
fig_prom = sns.barplot(y=precioPromedio.tipodepropiedad, x=precioPromedio.Precio_USD, data=precioPromedio, orient='h', palette = (sns.color_palette("viridis", 24)))

fig_prom.set_title("Precio Promedio según Tipo de Propiedad", fontsize = 15)

fig_prom.set_ylabel("", fontsize = 12)

fig_prom.set_xlabel("Precio Promedio (USD)", fontsize = 12)
orden = tipos.index
fig_box = sns.boxplot(y=precioPorTipo.tipodepropiedad, x=precioPorTipo.Precio_USD, orient='h',data=precioPorTipo, order=orden)

fig_box.set_title("Precio según Tipo de Propiedad", fontsize = 15)

fig_box.set_ylabel("", fontsize = 12)

fig_box.set_xlabel("Precio (USD)", fontsize = 12)
fig_scat = sns.stripplot(y='tipodepropiedad', x='Precio_USD', orient='h',data=precioPorTipo, jitter=True, size=1.3, order=orden)

fig_scat.set_title("Precio según Tipo de Propiedad", fontsize = 15)

fig_scat.set_ylabel("", fontsize = 12)

fig_scat.set_xlabel("Precio (USD)", fontsize = 12)