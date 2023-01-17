import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import geopandas

import seaborn as sns

plt.style.use('default') # para graficos matplotlib

plt.rcParams['figure.figsize'] = (10, 8)



sns.set(style="whitegrid") # grid seaborn



%config IPCompleter.greedy=True
df_props = pd.read_csv('train.csv')

df_props.info()
mexico_states = geopandas.read_file('mexstates.shp')
mexico_states.info()
mexico_states.head()
mexico_states = mexico_states.rename(columns = {'ADMIN_NAME': 'provincia'})
mexico_states.head()
# Determino si ambos dataframes contienen la misma cantidad de provincias



df_props.provincia.value_counts()
mexico_states.provincia.value_counts()
precio_por_provincia = df_props.groupby(by = 'provincia').precio.mean().sort_values(ascending = False)

precio_por_provincia
# Tengo que pasar los nombres de las provincias a de mis datos a los mismos que los de shapefile

# Para eso voy a ordenarlos a ambos alfabeticamente y ver que onda

dict(precio_por_provincia.sort_values(ascending = True))
dict(mexico_states.provincia.value_counts())
def unificar_nombre_provincias(provincia):

    if (provincia == 'Baja California Norte'):

        return 'Baja California'

    elif (provincia == 'Michoacán'):

        return 'Michoacan'

    elif (provincia == 'San luis Potosí'):

        return 'San Luis Potosi'

    elif (provincia == 'Yucatán'):

        return 'Yucatan'

    elif (provincia == 'Querétaro'):

        return 'Queretaro'

    elif (provincia == 'Nuevo León'):

        return 'Nuevo Leon'

    elif (provincia == 'Edo. de México'):

        return 'Mexico'

    

    return provincia
df_props.provincia = df_props.provincia.apply(unificar_nombre_provincias)
df_props.provincia.value_counts()
precio_por_provincia = df_props.groupby(by = 'provincia').precio.mean()
precio_por_provincia
mexico_states = mexico_states.join(precio_por_provincia, on = 'provincia')
mexico_states.info()
mexico_states.plot(column = 'precio', legend = True, cmap = 'OrRd')
plt.savefig('choropleth_precio_promedio_por_provincia.png')

plt.close()
mexico_states['precio_dolares'] = mexico_states['precio'].map(lambda x : x * 0.051)

mexico_states['precio_dolares']


# plot as usual, grab the axes 'ax' returned by the plot

colormap = "Reds"   # add _r to reverse the colormap

ax = mexico_states.plot(column='precio_dolares', cmap=colormap, \

                figsize=[12,9], \

                vmin=min(mexico_states.precio_dolares), vmax=max(mexico_states.precio_dolares))

plt.xlabel('Latitud', fontsize = 15)

plt.ylabel('Longitud', fontsize = 15)

# map marginal/face deco

ax.set_title('Precio USD por provincia', fontsize = 18)



# colorbar will be created by ...

fig = ax.get_figure()

# add colorbar axes to the figure

# here, need trial-and-error to get [l,b,w,h] right

# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)

cbax = fig.add_axes([0.95, 0.21, 0.03, 0.59])   



cbax.set_title('Precio USD')

sm = plt.cm.ScalarMappable(cmap=colormap, \

                norm=plt.Normalize(vmin=min(mexico_states.precio_dolares), vmax=max(mexico_states.precio_dolares)))

# at this stage, 

# 'cbax' is just a blank axes, with un needed labels on x and y axes



# blank-out the array of the scalar mappable 'sm'

sm._A = []

# draw colorbar into 'cbax'

fig.colorbar(sm, cax=cbax, format="%d")



# dont use: plt.tight_layout()

plt.show()
mexico_states.sort_values(by = 'precio_dolares', ascending = False)[['provincia', 'precio_dolares']]
plt.xticks(rotation = 'vertical')

plt.bar(x = 'provincia', height = 'precio_dolares', width = 0.8, data = mexico_states.sort_values(by = 'precio_dolares', ascending = True))
mexico_states['precio_dolares'].hist()
# Filtro las filas que tengan coordenadas validas

df_coordenadas = df_props[df_props['lat'].notnull() & df_props['lng'].notnull()]

df_coordenadas.head()
plt.scatter(x = 'lng',y = 'lat',data = df_coordenadas)
df_props.columns
df_props.info()
# Observo la antiguedad por provincia



antiguedad_por_provincia = df_props.groupby(by = 'provincia').antiguedad.mean().sort_values(ascending = True)

antiguedad_por_provincia
antiguedad_por_provincia.hist()
antiguedad_por_provincia.mean()
df_props.head()
mexico_states = mexico_states.join(antiguedad_por_provincia, on = 'provincia')
mexico_states.head()
mexico_states.plot(column = 'antiguedad', legend = True)
df_props.provincia.value_counts()
df_props.info()
plt.scatter(x= 'metroscubiertos',y = 'metrostotales',data = df_props)
# Analizar las propiedades mas grandes por provincia

# Analizar el valor/metroCuadrado por provincia

df_props.head()
df_props.tipodepropiedad.value_counts()
df_props['antiguedad'].value_counts()
# Cantidad de publicaciones por provincias

publicaciones_por_provincia = df_props.provincia.value_counts()

publicaciones_por_provincia = pd.DataFrame(data = publicaciones_por_provincia)

publicaciones_por_provincia = publicaciones_por_provincia.reset_index()

publicaciones_por_provincia = publicaciones_por_provincia.rename(columns = {'provincia': 'publicaciones', 'index': 'provincia'})

publicaciones_por_provincia = pd.DataFrame(publicaciones_por_provincia)
df_publicaciones = mexico_states.sort_values(by = 'provincia')

df_publicaciones = df_publicaciones.reset_index(drop = True)

publicaciones_por_provincia = publicaciones_por_provincia.sort_values(by = 'provincia')

publicaciones_por_provincia = publicaciones_por_provincia.reset_index(drop = True)

df_publicaciones['publicaciones'] = publicaciones_por_provincia['publicaciones']

df_publicaciones[['provincia', 'publicaciones']]

df_publicaciones.plot(column = 'publicaciones', cmap='OrRd', legend = True)
# plot as usual, grab the axes 'ax' returned by the plot

colormap = "Reds"   # add _r to reverse the colormap

ax = df_publicaciones.plot(column='publicaciones', cmap=colormap, \

                figsize=[12,9], \

                vmin=min(df_publicaciones.publicaciones), vmax=max(df_publicaciones.publicaciones))

plt.xlabel('Latitud', fontsize = 15)

plt.ylabel('Longitud', fontsize = 15)

# map marginal/face deco

ax.set_title('Publicaciones por provincia', fontsize = 18)



# colorbar will be created by ...

fig = ax.get_figure()

# add colorbar axes to the figure

# here, need trial-and-error to get [l,b,w,h] right

# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)

cbax = fig.add_axes([0.95, 0.21, 0.03, 0.59])   



cbax.set_title('Publicaciones')

sm = plt.cm.ScalarMappable(cmap=colormap, \

                norm=plt.Normalize(vmin=min(df_publicaciones.publicaciones), vmax=max(df_publicaciones.publicaciones)))

# at this stage, 

# 'cbax' is just a blank axes, with un needed labels on x and y axes



# blank-out the array of the scalar mappable 'sm'

sm._A = []

# draw colorbar into 'cbax'

fig.colorbar(sm, cax=cbax, format="%d")



# dont use: plt.tight_layout()

plt.show()
df_metrostotales = df_props[(df_props['metrostotales'].notnull())].groupby(by = 'provincia').metrostotales.mean()
mexico_states = mexico_states.join(df_metrostotales, on = 'provincia')
mexico_states.metrostotales.hist()
# plot as usual, grab the axes 'ax' returned by the plot

colormap = "Blues"   # add _r to reverse the colormap

ax = mexico_states.plot(column='metrostotales', cmap=colormap, \

                figsize=[12,9], \

                vmin=min(mexico_states.metrostotales), vmax=max(mexico_states.metrostotales))

plt.xlabel('Latitud', fontsize = 15)

plt.ylabel('Longitud', fontsize = 15)

# map marginal/face deco

ax.set_title('Metros totales por provincia', fontsize = 18)



# colorbar will be created by ...

fig = ax.get_figure()

# add colorbar axes to the figure

# here, need trial-and-error to get [l,b,w,h] right

# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)

cbax = fig.add_axes([0.95, 0.21, 0.03, 0.59])   



cbax.set_title('Metros')

sm = plt.cm.ScalarMappable(cmap=colormap, \

                norm=plt.Normalize(vmin=min(mexico_states.metrostotales), vmax=max(mexico_states.metrostotales)))

# at this stage, 

# 'cbax' is just a blank axes, with un needed labels on x and y axes



# blank-out the array of the scalar mappable 'sm'

sm._A = []

# draw colorbar into 'cbax'

fig.colorbar(sm, cax=cbax, format="%d")



# dont use: plt.tight_layout()

plt.show()

# plot as usual, grab the axes 'ax' returned by the plot

colormap = "Greens"   # add _r to reverse the colormap

ax = mexico_states.plot(column='antiguedad', cmap=colormap, \

                figsize=[12,9], \

                vmin=min(mexico_states.antiguedad), vmax=max(mexico_states.antiguedad))

plt.xlabel('Latitud', fontsize = 15)

plt.ylabel('Longitud', fontsize = 15)

# map marginal/face deco

ax.set_title('Antiguedad por provincia', fontsize = 18)



# colorbar will be created by ...

fig = ax.get_figure()

# add colorbar axes to the figure

# here, need trial-and-error to get [l,b,w,h] right

# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)

cbax = fig.add_axes([0.95, 0.21, 0.03, 0.59])   



cbax.set_title('Antiguedad')

sm = plt.cm.ScalarMappable(cmap=colormap, \

                norm=plt.Normalize(vmin=min(mexico_states.antiguedad), vmax=max(mexico_states.antiguedad)))

# at this stage, 

# 'cbax' is just a blank axes, with un needed labels on x and y axes



# blank-out the array of the scalar mappable 'sm'

sm._A = []

# draw colorbar into 'cbax'

fig.colorbar(sm, cax=cbax, format="%d")



# dont use: plt.tight_layout()

plt.show()
df_props.antiguedad.value_counts()
df_metrostotales = df_props[df_props.metrostotales.notnull()]

df_metrostotales.info()
df_props.groupby(by = 'provincia').metrostotales.mean().sort_values()
# plot as usual, grab the axes 'ax' returned by the plot

colormap = "Greens"   # add _r to reverse the colormap

ax = mexico_states.plot(column='metrostotales', cmap=colormap, \

                figsize=[12,9], \

                vmin=min(mexico_states.metrostotales), vmax=max(mexico_states.metrostotales))

plt.xlabel('Latitud', fontsize = 15)

plt.ylabel('Longitud', fontsize = 15)

# map marginal/face deco

ax.set_title('Metros por provincia', fontsize = 18)



# colorbar will be created by ...

fig = ax.get_figure()

# add colorbar axes to the figure

# here, need trial-and-error to get [l,b,w,h] right

# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)

cbax = fig.add_axes([0.95, 0.21, 0.03, 0.59])   



cbax.set_title('Metros')

sm = plt.cm.ScalarMappable(cmap=colormap, \

                norm=plt.Normalize(vmin=min(mexico_states.metrostotales), vmax=max(mexico_states.metrostotales)))

# at this stage, 

# 'cbax' is just a blank axes, with un needed labels on x and y axes



# blank-out the array of the scalar mappable 'sm'

sm._A = []

# draw colorbar into 'cbax'

fig.colorbar(sm, cax=cbax, format="%d")



# dont use: plt.tight_layout()

plt.show()
df_metrostotales.hist()
plt.xticks(rotation = 'vertical')

sns.barplot(data=publicaciones_por_provincia.sort_values(by = 'publicaciones', ascending = False), x='provincia',y='publicaciones', orient='v', palette = (sns.color_palette("viridis",)))

plt.title('Publicaciones por provincia', fontsize = 15)

plt.ylabel('Publicaciones', fontsize = 12)

plt.xlabel('Provincia', fontsize = 12)

#publicaciones_por_provincia
mexico_states.info()
mexico_states['precio_metro_dolares'] = mexico_states['precio_dolares'] / mexico_states['metrostotales']
mexico_states[['provincia', 'precio_metro_dolares']]
# plot as usual, grab the axes 'ax' returned by the plot

colormap = "Reds"   # add _r to reverse the colormap

ax = mexico_states.plot(column='precio_metro_dolares', cmap=colormap, \

                figsize=[12,9], \

                vmin=min(mexico_states.precio_metro_dolares), vmax=max(mexico_states.precio_metro_dolares))

plt.xlabel('Latitud', fontsize = 15)

plt.ylabel('Longitud', fontsize = 15)

# map marginal/face deco

ax.set_title('Precio/metro USD por provincia', fontsize = 18)



# colorbar will be created by ...

fig = ax.get_figure()

# add colorbar axes to the figure

# here, need trial-and-error to get [l,b,w,h] right

# l:left, b:bottom, w:width, h:height; in normalized unit (0-1)

cbax = fig.add_axes([0.95, 0.21, 0.03, 0.59])   



cbax.set_title('Precio USD')

sm = plt.cm.ScalarMappable(cmap=colormap, \

                norm=plt.Normalize(vmin=min(mexico_states.precio_metro_dolares), vmax=max(mexico_states.precio_metro_dolares)))

# at this stage, 

# 'cbax' is just a blank axes, with un needed labels on x and y axes



# blank-out the array of the scalar mappable 'sm'

sm._A = []

# draw colorbar into 'cbax'

fig.colorbar(sm, cax=cbax, format="%d")



# dont use: plt.tight_layout()

plt.show()
plt.xticks(rotation = 'vertical')

df_yucatan = df_props[df_props.provincia == 'Yucatan']

ax = sns.countplot(x="tipodepropiedad", data=df_yucatan, order = df_yucatan['tipodepropiedad'].value_counts().index)

plt.title('Tipo de propiedades en Yucatan')

plt.ylabel('Publicaciones', fontsize = 12)

plt.xlabel('Tipo de propiedad', fontsize = 12)

#publicaciones_por_provincia
plt.xticks(rotation = 'vertical')

df_yucatan = df_props[df_props.provincia == 'Baja California']

ax = sns.countplot(x="tipodepropiedad", data=df_yucatan, order = df_yucatan['tipodepropiedad'].value_counts().index)

plt.title('Tipo de propiedades en Baja California Norte')

plt.ylabel('Publicaciones', fontsize = 12)

plt.xlabel('Tipo de propiedad', fontsize = 12)

#publicaciones_por_provincia