# Problemas al correr, no llega el código de verificación para activar internet e instalar los componentes necesarios



import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import numpy as np

import seaborn as sns

from math import pi

import geopandas as gp

import adjustText as aT



zonaProp = pd.read_csv('./data/train.csv')
zonaProp['piscina'] = zonaProp['piscina'].astype('category')

zonaProp['usosmultiples'] = zonaProp['usosmultiples'].astype('category')

zonaProp['escuelascercanas'] = zonaProp['escuelascercanas'].astype('category')

zonaProp['centroscomercialescercanos'] = zonaProp['centroscomercialescercanos'].astype('category')

zonaProp['gimnasio'] = zonaProp['gimnasio'].astype('category')

zonaProp['banos'] = zonaProp['banos'].astype('float16')

zonaProp['garages'] = zonaProp['garages'].astype('float16')

zonaProp['habitaciones'] = zonaProp['habitaciones'].astype('float16')

zonaProp['antiguedad'] = zonaProp['antiguedad'].astype('float16')



zonaProp['fecha'] = pd.to_datetime(zonaProp['fecha'])



zonaProp["ano"] = zonaProp["fecha"].dt.year

zonaProp["mes"] = zonaProp["fecha"].dt.month

zonaProp["dia"] = zonaProp["fecha"].dt.day
import plotly.graph_objects as go



zonaCostero = zonaProp.loc[(zonaProp["ciudad"] == "Acapulco de Juárez") | (zonaProp["ciudad"] == "Zihuatanejo de Azueta")]

zonaNoCostero = zonaProp.loc[(zonaProp["ciudad"] == "Uruapan") | (zonaProp["ciudad"] == "Tarímbaro") | (zonaProp["ciudad"] == "Morelia") | (zonaProp["ciudad"] == "Chilapa de Alvarez") | (zonaProp["ciudad"] == "Iguala de la Independencia") | (zonaProp["ciudad"] == "Chilpancingo de los Bravo") | (zonaProp["ciudad"] == "Coyuca de Benítez") | (zonaProp["ciudad"] == "ZirAndaro") | (zonaProp["ciudad"] == "Copala") | (zonaProp["ciudad"] == "Técpan de Galeana") | (zonaProp["ciudad"] == "Chilapa de Alvarez") | (zonaProp["ciudad"] == "Atoyac de Alvarez")]



zonaCosteraPropiedad = zonaCostero.groupby(["piscina", "tipodepropiedad"]).agg({"id":"count"}).reset_index()

zonaNoCosteraPropiedad = zonaNoCostero.groupby(["piscina", "tipodepropiedad"]).agg({"id":"count"}).reset_index()



cant_costeros = zonaCostero["id"].count()

ciudad_costera = f"Ciudad costera: {cant_costeros}"

cant_no_costeros = zonaNoCostero["id"].count()

ciudad_no_costera = f"Ciudad NO costera: {cant_no_costeros}"



zonaCosteraPropiedad1 = zonaCosteraPropiedad.loc[zonaCosteraPropiedad["piscina"] == 1.0]

zonaNoCosteraPropiedad1 = zonaNoCosteraPropiedad.loc[zonaNoCosteraPropiedad["piscina"] == 1.0]

zonaCosteraPropiedad2 = zonaCosteraPropiedad.loc[zonaCosteraPropiedad["piscina"] == 0.0]

zonaNoCosteraPropiedad2 = zonaNoCosteraPropiedad.loc[zonaNoCosteraPropiedad["piscina"] == 0.0]



cant_costeros_con_piscina = zonaCosteraPropiedad1["id"].sum()

cant_costeros_sin_piscina = cant_costeros - cant_costeros_con_piscina



cant_no_costeros_con_piscina = zonaNoCosteraPropiedad1["id"].sum()

cant_no_costeros_sin_piscina = cant_no_costeros - cant_no_costeros_con_piscina



tiene_piscina = f"Tiene piscina: {cant_costeros_con_piscina + cant_no_costeros_con_piscina}"

no_tiene_piscina = f"NO tiene piscina: {cant_costeros_sin_piscina + cant_no_costeros_sin_piscina}"



cant_costera_piscina_casa = zonaCosteraPropiedad1.loc[zonaCosteraPropiedad1["tipodepropiedad"] == "Casa"]["id"].sum()

cant_no_costera_piscina_casa = zonaNoCosteraPropiedad1.loc[zonaNoCosteraPropiedad1["tipodepropiedad"] == "Casa"]["id"].sum()

cant_con_piscina_casa = cant_costera_piscina_casa + cant_no_costera_piscina_casa



cant_costera_piscina_apartamento = zonaCosteraPropiedad1.loc[zonaCosteraPropiedad1["tipodepropiedad"] == "Apartamento"]["id"].sum()

cant_no_costera_piscina_apartamento = zonaNoCosteraPropiedad1.loc[zonaNoCosteraPropiedad1["tipodepropiedad"] == "Apartamento"]["id"].sum()

cant_con_piscina_apartamento = cant_costera_piscina_apartamento + cant_no_costera_piscina_apartamento



cant_pileta = cant_costeros_con_piscina + cant_no_costeros_con_piscina

cant_costera_piscina_otro = cant_pileta - cant_con_piscina_casa - cant_con_piscina_apartamento



cant_costera_sin_piscina_casa = zonaCosteraPropiedad2.loc[zonaCosteraPropiedad2["tipodepropiedad"] == "Casa"]["id"].sum()

cant_no_costera_sin_piscina_casa = zonaNoCosteraPropiedad2.loc[zonaNoCosteraPropiedad2["tipodepropiedad"] == "Casa"]["id"].sum()

cant_sin_piscina_casa = cant_costera_sin_piscina_casa + cant_no_costera_sin_piscina_casa



cant_costera_sin_piscina_apartamento = zonaCosteraPropiedad2.loc[zonaCosteraPropiedad2["tipodepropiedad"] == "Apartamento"]["id"].sum()

cant_no_costera_sin_piscina_apartamento = zonaNoCosteraPropiedad2.loc[zonaNoCosteraPropiedad2["tipodepropiedad"] == "Apartamento"]["id"].sum()

cant_sin_piscina_apartamento = cant_costera_sin_piscina_apartamento + cant_no_costera_sin_piscina_apartamento



cant_sin_pileta = cant_costeros_sin_piscina + cant_no_costeros_sin_piscina

cant_costera_sin_piscina_otro = cant_sin_pileta - cant_sin_piscina_casa - cant_sin_piscina_apartamento



fig = go.Figure(data=[go.Sankey(

    node = dict(

      pad = 15,

      thickness = 20,

      line = dict(color = "black", width = 0.5),

      label = ["Ciudad costera", "Ciudad NO costera", "Tiene piscina", "NO tiene piscina", "Casa","Apartamento", "Otros"],

      color = "blue"

    ),

    link = dict(

      source = [0, 0, 1, 1, 2, 2, 2, 3, 3, 3],

      target = [2, 3, 2, 3, 4, 5, 6, 4, 5, 6],

      value = [cant_costeros_con_piscina, cant_costeros_sin_piscina, cant_no_costeros_con_piscina, cant_no_costeros_sin_piscina, cant_con_piscina_casa, cant_con_piscina_apartamento, cant_costera_piscina_otro, cant_sin_piscina_casa, cant_sin_piscina_apartamento, cant_costera_sin_piscina_otro]

  ))])



fig.update_layout(title_text="Relación entre cercanía costera y piscinas", font_size=10)

fig.show()
zonaCasa = zonaProp.loc[(zonaProp["tipodepropiedad"] == "Apartamento") | (zonaProp["tipodepropiedad"] == "Casa")]

plt.figure(figsize=(15, 5))

ax = sns.lineplot(x="antiguedad", y="precio", hue="tipodepropiedad", data=zonaCasa)
zonaCasa = zonaProp.loc[(zonaProp["tipodepropiedad"] == "Apartamento") | (zonaProp["tipodepropiedad"] == "Casa")]

plt.figure(figsize=(15, 5))

ax = sns.lineplot(x="ano", y="precio", hue="tipodepropiedad", data=zonaCasa)
zonaCasa = zonaProp.loc[zonaProp["tipodepropiedad"] == "Casa"]

plt.figure(figsize=(15, 5))

ax = sns.lineplot(x="ano", y="precio", hue="banos", data=zonaCasa)
zonaCasa = zonaProp.loc[zonaProp["tipodepropiedad"] == "Apartamento"]

plt.figure(figsize=(15, 5))

ax = sns.lineplot(x="ano", y="precio", hue="banos", data=zonaCasa)
zonaCasa = zonaProp.loc[zonaProp["tipodepropiedad"] == "Casa"]

plt.figure(figsize=(15, 5))

ax = sns.lineplot(x="ano", y="precio", hue="garages", data=zonaCasa)
zonaApartamento = zonaProp.loc[zonaProp["tipodepropiedad"] == "Apartamento"]

plt.figure(figsize=(15, 5))

ax = sns.lineplot(x="ano", y="precio", hue="garages", data=zonaApartamento)