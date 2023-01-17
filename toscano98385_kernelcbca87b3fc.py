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
piscina = df_props_full.loc[:,['piscina','tipodepropiedad','Precio_MEX','Precio_USD']]

# Verifico que tipo de propiedades suelen tener piscina para filtrar datos

print(len(piscina))

piscina.loc[piscina.piscina == 1].tipodepropiedad.value_counts()
tiposConPiscina = piscina.loc[piscina.piscina == 1].tipodepropiedad.value_counts().index.array

tiposConPiscina
# Descarto las publicaciones que no pertenecen a tipos con piscina

piscina = piscina[piscina.tipodepropiedad.apply(lambda x: x in tiposConPiscina)]

piscina['piscina'] = piscina['piscina'].apply(lambda x: 'Con Piscina' if x == 1 else 'Sin Piscina')

print(len(piscina))

piscina.head()
sns.boxplot(x=piscina.piscina, y=piscina.Precio_USD, data=piscina, width=0.9)

plt.title('Variación Precio con Piscina vs sin Piscina', fontsize = 15)

plt.ylabel('Precio (USD)', fontsize = 12)

plt.xlabel('', fontsize = 12)
usosmultiples = df_props_full.loc[:,['usosmultiples','tipodepropiedad','Precio_MEX','Precio_USD']]

# Verifico que tipo de propiedades suelen tener usosmultiples para filtrar datos

usosmultiples.loc[usosmultiples.usosmultiples == 1].tipodepropiedad.value_counts()
tiposConUsos = usosmultiples.loc[usosmultiples.usosmultiples == 1].tipodepropiedad.value_counts().index.array

tiposConUsos
# Descarto las publicaciones que no pertenecen a tipos con piscina

usosmultiples = usosmultiples[usosmultiples.tipodepropiedad.apply(lambda x: x in tiposConUsos)]

usosmultiples['usosmultiples'] = usosmultiples['usosmultiples'].apply(lambda x: 'Con Usos Múltiples' if x == 1 else 'Sin Usos Múltiples')

print(len(usosmultiples))

usosmultiples.head()
sns.boxplot(x=usosmultiples.usosmultiples, y=usosmultiples.Precio_USD, data=usosmultiples, width=0.9)

plt.title('Variación Precio con Usos Múltiples vs sin Usos Múltiples', fontsize = 15)

plt.ylabel('Precio (USD)', fontsize = 12)

plt.xlabel('', fontsize = 12)
gimnasio = df_props_full.loc[:,['gimnasio','tipodepropiedad','Precio_MEX','Precio_USD']]

# Verifico que tipo de propiedades suelen tener gimnasio para filtrar datos

print(len(gimnasio))

gimnasio.loc[gimnasio.gimnasio == 1].tipodepropiedad.value_counts()
tiposConGimnasio = gimnasio.loc[gimnasio.gimnasio == 1].tipodepropiedad.value_counts().index.array

tiposConGimnasio
# Descarto las publicaciones que no pertenecen a tipos con gimnasio

gimnasio = gimnasio[gimnasio.tipodepropiedad.apply(lambda x: x in tiposConGimnasio)]

gimnasio['gimnasio'] = gimnasio['gimnasio'].apply(lambda x: 'Con Gimnasio' if x == 1 else 'Sin Gimnasio')

print(len(gimnasio))

gimnasio.head()
sns.boxplot(x=gimnasio.gimnasio, y=gimnasio.Precio_USD, data=gimnasio, width=0.9)

plt.title('Variación Precio con Gimnasio vs sin Gimnasio', fontsize = 15)

plt.ylabel('Precio (USD)', fontsize = 12)

plt.xlabel('', fontsize = 12)
servicios = df_props_full.loc[:,['piscina','gimnasio','usosmultiples','tipodepropiedad','Precio_MEX','Precio_USD']]

# Verifico que tipo de propiedades suelen tener piscina y/o gimnasio y/o SUM para filtrar datos

print(len(servicios))

servicios.loc[(servicios.piscina == 1) | (servicios.gimnasio == 1) | (servicios.usosmultiples == 1)].tipodepropiedad.value_counts()
servicios.head()
tiposConServicios = servicios.loc[(servicios.piscina == 1) | (servicios.gimnasio == 1) | (servicios.usosmultiples == 1)].tipodepropiedad.value_counts().index.array

tiposConServicios
def serviciosDisponibles(row):

    # row[0] == 'piscina'  - row[1] == 'gimnasio' - row[2] == 'usosmultiples'

    if ((row[0] == 1) & (row[1]==1) & (row[2]==1)):

        return 'Todos'

    if ((row[0] == 1) & (row[1]==1)):

        return 'Piscina y Gimnasio'

    if ((row[0] == 1) & (row[2]==1)):

        return 'Piscina y SUM'

    if ((row[2] == 1) & (row[1]==1)):

        return 'SUM y Gimnasio'

    if (row[0] == 1):

        return 'Piscina'

    if (row[1] == 1):

        return 'Gimnasio'

    if (row[2] == 1):

        return 'SUM'

    return 'Ninguno'
# Descarto las publicaciones que no pertenecen a tipos con servicios

servicios = servicios[servicios.tipodepropiedad.apply(lambda x: x in tiposConServicios)]

servicios['servicios'] = servicios.apply(serviciosDisponibles, axis=1)

print(len(servicios))

servicios.head()
servicios.servicios.value_counts()
srv_fig = sns.boxplot(x=servicios.servicios, y=servicios.Precio_USD, data=servicios, width=0.9, \

            order=['Ninguno','Gimnasio','Piscina','SUM','Piscina y SUM','SUM y Gimnasio','Piscina y Gimnasio','Todos'])

srv_fig.set_xticklabels(srv_fig.get_xticklabels(), rotation=30, ha="right")





plt.title('Variación Precio según Servicios Disponibles', fontsize = 15)

plt.ylabel('Precio (USD)', fontsize = 12)

plt.xlabel('', fontsize = 12)
serv = servicios.servicios.value_counts().to_frame().reset_index()

serv.columns = ['prop','serv']



srv_bar = sns.barplot(data=serv, x='prop',y='serv', orient='v', palette = (sns.color_palette("viridis",)))

srv_bar.set_xticklabels(srv_bar.get_xticklabels(), rotation=30, ha="right")



plt.title('Cantidad Publicaciones según Servicios', fontsize = 15)

plt.ylabel('Cantidad Publicaciones', fontsize = 12)

plt.xlabel('Servicio Ofrecido', fontsize = 12)
escuelascercanas = df_props_full.loc[:,['escuelascercanas','tipodepropiedad','Precio_MEX','Precio_USD']]

# Verifico que tipo de propiedades suelen tener escuelascercanas para filtrar datos

print(len(escuelascercanas))

escuelascercanas.loc[escuelascercanas.escuelascercanas == 1].tipodepropiedad.value_counts()
tiposConEscuelasCercanas = escuelascercanas.loc[escuelascercanas.escuelascercanas == 1].tipodepropiedad.value_counts().index.array

tiposConEscuelasCercanas
# Descarto las publicaciones que no pertenecen a tipos con escuelascercanas

escuelascercanas = escuelascercanas[escuelascercanas.tipodepropiedad.apply(lambda x: x in tiposConEscuelasCercanas)]

escuelascercanas['escuelascercanas'] = escuelascercanas['escuelascercanas'].apply(lambda x: 'Con Escuelas Cercanas' if x == 1 else 'Sin Escuelas Cercanas')

print(len(escuelascercanas))

escuelascercanas.head()
sns.boxplot(x=escuelascercanas.escuelascercanas, y=escuelascercanas.Precio_USD, data=escuelascercanas, width=0.9, order=['Sin Escuelas Cercanas','Con Escuelas Cercanas'])

plt.title('Variación Precio con Escuelas Cercanas vs sin Escuelas Cercanas', fontsize = 15)

plt.ylabel('Precio (USD)', fontsize = 12)

plt.xlabel('', fontsize = 12)
centroscercanos = df_props_full.loc[:,['centroscomercialescercanos','tipodepropiedad','Precio_MEX','Precio_USD']]

# Verifico que tipo de propiedades suelen tener centroscomercialescercanos para filtrar datos

print(len(centroscercanos))

centroscercanos.loc[centroscercanos.centroscomercialescercanos == 1].tipodepropiedad.value_counts()
tiposConCentrosCercanos = centroscercanos.loc[centroscercanos.centroscomercialescercanos == 1].tipodepropiedad.value_counts().index.array

tiposConCentrosCercanos
# Descarto las publicaciones que no pertenecen a tipos con escuelascercanas

centroscercanos = centroscercanos[centroscercanos.tipodepropiedad.apply(lambda x: x in tiposConCentrosCercanos)]

centroscercanos['centroscomercialescercanos'] = centroscercanos['centroscomercialescercanos'].apply(lambda x: 'Con Centros Comerciales Cercanos' if x == 1 else 'Sin Centros Comerciales Cercanos')

print(len(centroscercanos))

centroscercanos.head()
sns.boxplot(x=centroscercanos.centroscomercialescercanos, y=centroscercanos.Precio_USD, data=centroscercanos, width=0.9, order=['Sin Centros Comerciales Cercanos','Con Centros Comerciales Cercanos'])

plt.title('Variación Precio con Centros Comerciales Cercanos vs sin Centros Comerciales Cercanos', fontsize = 15)

plt.ylabel('Precio (USD)', fontsize = 12)

plt.xlabel('', fontsize = 12)
fig1, axes1 = plt.subplots(1,2, sharey='row')

sns.boxplot(ax=axes1[0], x=escuelascercanas.escuelascercanas, y=escuelascercanas.Precio_USD, data=escuelascercanas, width=0.9, order=['Sin Escuelas Cercanas','Con Escuelas Cercanas'])

sns.boxplot(ax=axes1[1], x=centroscercanos.centroscomercialescercanos, y=centroscercanos.Precio_USD, data=centroscercanos, width=0.9, order=['Sin Centros Comerciales Cercanos','Con Centros Comerciales Cercanos'])



axes1[0].set_xlabel('')

axes1[1].set_ylabel('Precio (USD)', fontsize = 12)



axes1[1].set_xlabel('')

axes1[1].set_ylabel('')



fig1.suptitle('Precio según Propiedades Cercanas', fontsize = 15)

# fig1.supylabel('Precio (USD)', fontsize = 12)

# fig1.supxlabel('', fontsize = 12)

plt.show()
escuelasYCentros = df_props_full.loc[:,['centroscomercialescercanos','escuelascercanas','tipodepropiedad','Precio_MEX','Precio_USD']]

# Verifico que tipo de propiedades suelen tener centroscomercialescercanos y/o escuelascercanas para filtrar datos

print(len(escuelasYCentros))

escuelasYCentros.loc[(escuelasYCentros.centroscomercialescercanos == 1) | (escuelasYCentros.escuelascercanas == 1)].tipodepropiedad.value_counts()
escuelasYCentros.head()
tiposConPropsCercanas = escuelasYCentros.loc[(escuelasYCentros.centroscomercialescercanos == 1) | (escuelasYCentros.escuelascercanas == 1)].tipodepropiedad.value_counts().index.array

tiposConPropsCercanas
def propiedadCercana(row):

    # row[0] == 'centroscomercialescercanos'  - row[1] == 'escuelascercanas'

    if ((row[0] == 1) & (row[1]==1)):

        return 'Ambos'

    if (row[0] == 1):

        return 'Centros Comerciales'

    if (row[1] == 1):

        return 'Escuelas'

    return 'Ninguno'
# Descarto las publicaciones que no pertenecen a tipos con escuelascercanas

escuelasYCentros = escuelasYCentros[escuelasYCentros.tipodepropiedad.apply(lambda x: x in tiposConCentrosCercanos)]

escuelasYCentros['cercanos'] = escuelasYCentros.apply(propiedadCercana, axis=1)

print(len(escuelasYCentros))

escuelasYCentros.head()
escuelasYCentros.cercanos.value_counts()
sns.boxplot(x=escuelasYCentros.cercanos, y=escuelasYCentros.Precio_USD, data=escuelasYCentros, width=0.9, order=['Ninguno','Escuelas','Centros Comerciales','Ambos'])

plt.title('Variación Precio según Propiedades Cercanas', fontsize = 15)

plt.ylabel('Precio (USD)', fontsize = 12)

plt.xlabel('', fontsize = 12)
cerc = escuelasYCentros.cercanos.value_counts().to_frame().reset_index()

cerc.columns = ['prop','pubs']

sns.barplot(data=cerc, x='prop',y='pubs', orient='v', palette = (sns.color_palette("viridis",)))

plt.title('Cantidad Publicaciones según Propiedades Cercanas', fontsize = 15)

plt.ylabel('Cantidad Publicaciones', fontsize = 12)

plt.xlabel('Propiedades Cercanas', fontsize = 12)
pubsPorTipoYCaracteristica = df_props_full.groupby('tipodepropiedad').agg({'piscina':'sum','gimnasio':'sum','usosmultiples':'sum','escuelascercanas':'sum','centroscomercialescercanos':'sum'})

pubsPorTipoYCaracteristica
sns.heatmap(pubsPorTipoYCaracteristica, cmap='Spectral', annot = False)