# Problemas al correr, no llega el código de verificación para activar internet e instalar los componentes necesarios



import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import numpy as np

import seaborn as sns

from math import pi

import geopandas as gp

import adjustText as aT

from pathlib import Path



zonaProp = pd.read_csv('./data/train.csv')
import re

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

import nltk

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

nltk.download('tagsets')

nltk.download('cess_esp')
## Funcion auxiliar para filtrar palabras perteneciente a la categoría pasada



def filtrado_nltk(arr_tupla, categoria):

    arr_resultado = []

    for tupla in arr_tupla:

        if tupla[1] == categoria:

            arr_resultado.append(tupla[0])

    return arr_resultado



def filtrado_nltk_arr(tupla_arr, categoria_arr):

    arr_resultado = []

    for tupla in tupla_arr:

        if tupla[1] in categoria_arr:

            arr_resultado.append(tupla[0])

    return arr_resultado
# Corrección del encoding, y filtrado de elementos inncesarios en los textos



def sanitize(texto):

    palabras = []

    aux = texto

    if str(aux) == 'nan':

        return []

    aux = re.sub('&aacute;','á', aux)

    aux = re.sub('&aacute;','á', aux)

    aux = re.sub('&Agrave;', 'À', aux)

    aux = re.sub('&Aacute;', 'Á', aux)

    aux = re.sub('&Acirc;', 'Â', aux)

    aux = re.sub('&Atilde;', 'Ã', aux)

    aux = re.sub('&Auml;', 'Ä', aux)

    aux = re.sub('&Aring;', 'Å', aux)

    aux = re.sub('&agrave;', 'à', aux)

    aux = re.sub('&aacute;', 'á', aux)

    aux = re.sub('&acirc;', 'â', aux)

    aux = re.sub('&atilde;', 'ã', aux)

    aux = re.sub('&auml;', 'ä', aux)

    aux = re.sub('&aring;', 'å', aux)

    aux = re.sub('&AElig;', 'Æ', aux)

    aux = re.sub('&aelig;', 'æ', aux)

    aux = re.sub('&szlig;', 'ß', aux)

    aux = re.sub('&Ccedil;', 'Ç', aux)

    aux = re.sub('&ccedil;', 'ç', aux)

    aux = re.sub('&Egrave;', 'È', aux)

    aux = re.sub('&Eacute;', 'É', aux)

    aux = re.sub('&Ecirc;', 'Ê', aux)

    aux = re.sub('&Euml;', 'Ë', aux)

    aux = re.sub('&egrave;', 'è', aux)

    aux = re.sub('&eacute;', 'é', aux)

    aux = re.sub('&ecirc;', 'ê', aux)

    aux = re.sub('&euml;', 'ë', aux)

    aux = re.sub('&#131;', 'ƒ', aux)

    aux = re.sub('&Igrave;', 'Ì', aux)

    aux = re.sub('&Iacute;', 'Í', aux)

    aux = re.sub('&Icirc;', 'Î', aux)

    aux = re.sub('&Iuml;', 'Ï', aux)

    aux = re.sub('&igrave;', 'ì', aux)

    aux = re.sub('&iacute;', 'í', aux)

    aux = re.sub('&icirc;', 'î', aux)

    aux = re.sub('&iuml;', 'ï', aux)

    aux = re.sub('&Ntilde;', 'Ñ', aux)

    aux = re.sub('&ntilde;', 'ñ', aux)

    aux = re.sub('&Ograve;', 'Ò', aux)

    aux = re.sub('&Oacute;', 'Ó', aux)

    aux = re.sub('&Ocirc;', 'Ô', aux)

    aux = re.sub('&Otilde;', 'Õ', aux)

    aux = re.sub('&Ouml;', 'Ö', aux)

    aux = re.sub('&ograve;', 'ò', aux)

    aux = re.sub('&oacute;', 'ó', aux)

    aux = re.sub('&ocirc;', 'ô', aux)

    aux = re.sub('&otilde;', 'õ', aux)

    aux = re.sub('&ouml;', 'ö', aux)

    aux = re.sub('&Oslash;', 'Ø', aux)

    aux = re.sub('&oslash;', 'ø', aux)

    aux = re.sub('&#140;', 'Œ', aux)

    aux = re.sub('&#156;', 'œ', aux)

    aux = re.sub('&#138;', 'Š', aux)

    aux = re.sub('&#154;', 'š', aux)

    aux = re.sub('&Ugrave;', 'Ù', aux)

    aux = re.sub('&Uacute;', 'Ú', aux)

    aux = re.sub('&Ucirc;', 'Û', aux)

    aux = re.sub('&Uuml;', 'Ü', aux)

    aux = re.sub('&ugrave;', 'ù', aux)

    aux = re.sub('&uacute;', 'ú', aux)

    aux = re.sub('&ucirc;', 'û', aux)

    aux = re.sub('&uuml;', 'ü', aux)

    aux = re.sub('&#181;', 'µ', aux)

    aux = re.sub('&#215;', '×', aux)

    aux = re.sub('&Yacute;', 'Ý', aux)

    aux = re.sub('&#159;', 'Ÿ', aux)

    aux = re.sub('&yacute;', 'ý', aux)

    aux = re.sub('&yuml;', 'ÿ', aux)

    aux = re.sub('&#176;', '°', aux)

    aux = re.sub('&#134;', '†', aux)

    aux = re.sub('&#135;', '‡', aux)

    aux = re.sub('&lt;', '<', aux)

    aux = re.sub('&gt;', '>', aux)

    aux = re.sub('&#177;', '±', aux)

    aux = re.sub('&#171;', '«', aux)

    aux = re.sub('&#187;', '»', aux)

    aux = re.sub('&#191;', '¿', aux)

    aux = re.sub('&#161;', '¡', aux)

    aux = re.sub('&#183;', '·', aux)

    aux = re.sub('&#149;', '•', aux)

    aux = re.sub('&#153;', '™', aux)

    aux = re.sub('&copy;', '©', aux)

    aux = re.sub('&reg;', '®', aux)

    aux = re.sub('&#167;', '§', aux)

    aux = re.sub('&#182;', '¶', aux)

    aux = re.sub('<.*?>','', aux)

    aux = re.sub('t.v.','tv', aux)

    aux = re.sub('\\x95','', aux)

    aux = re.sub('&nbsp;','', aux)

    

    palabras.extend(nltk.word_tokenize(aux))



#     i = 0

#     for palabra in palabras:

#         if palabra == '.':

#             del(palabras[i])

#             continue

#         i += 1



    return palabras
def coefDecorado(descripcion, contador):

    contador[0] = contador[0] + 1

    print(contador[0])

    if descripcion == 'nan' or descripcion == None:

        return 0

    if isinstance(descripcion, str) and len(descripcion) > 500:

        if contador[0] == 170: print("acaaaa")

        return 0

        descripcion = descripcion[0:500]

    palabras_split_arr = sanitize(descripcion)

    palabras_tag = pos_tagger.tag(palabras_split_arr)

    cantTotalPalabras = 0;

    cantTotalAdjetivos = 0

    puntuacion = ['f0', 'faa', 'fat', 'fc', 'fca', 'fct', 'fd', 'fe', 'fg', 'fh', 'fia', 'fit', 'fp', 'fpa', 'fpt', 'fra', 'frc', 'fs', 'ft', 'fx', 'fz']

    numeracion = ['z0', 'zu']

    preposiciones = ['sp000']

    adjetivos = ['ao0000', 'aq0000']

    for palabra_tupla in palabras_tag:

        if palabra_tupla[1] in puntuacion or palabra_tupla[1] in numeracion or palabra_tupla[1] in preposiciones:

            continue

        if palabra_tupla[1] in adjetivos:

            cantTotalAdjetivos += 1

        cantTotalPalabras += 1

    if cantTotalAdjetivos == 0 or cantTotalPalabras == 0:

        return 0

    return cantTotalAdjetivos/cantTotalPalabras
coefDecorado = pd.read_csv('./data/coefdecorado_df.csv')

coefDecorado = coefDecorado.loc[coefDecorado["factorDecoradoDescripcion"] != 0.0]

zonaPropMerge = pd.merge(zonaProp, coefDecorado, on='id', how='inner')
df1 = zonaPropMerge.loc[zonaProp["provincia"] == "Distrito Federal"] 

df1 = df1.loc[df1["ciudad"] != "otra"]

df1 = df1.loc[df1["ciudad"] != "Milpa Alta"]

df1 = df1.groupby("ciudad").agg({"precio":"mean", "piscina":"sum", "gimnasio":"sum", "metrostotales":"mean","habitaciones":"mean", "banos":"mean", "garages":"mean", "factorDecoradoDescripcion":"mean", "id":"count"}).reset_index().nlargest(16, "precio")

df1[['ciudad', 'precio', 'factorDecoradoDescripcion']].plot(x='ciudad', secondary_y='factorDecoradoDescripcion', kind = "line",figsize=(15, 7));
df1 = zonaPropMerge.loc[zonaProp["provincia"] == "Distrito Federal"] 

df1 = df1.loc[df1["ciudad"] != "otra"]

df1 = df1.loc[df1["ciudad"] != "Milpa Alta"]

df1 = df1.loc[df1["ciudad"] != "Iztacalco"]

df1 = df1.groupby("ciudad").agg({"precio":"mean", "piscina":"sum", "gimnasio":"sum", "metrostotales":"mean","habitaciones":"mean", "banos":"mean", "garages":"mean", "factorDecoradoDescripcion":"mean", "id":"count"}).reset_index().nlargest(16, "precio")



df2 = df1[['ciudad', 'precio', 'factorDecoradoDescripcion']]



fig, ax = plt.subplots(figsize=(20, 8))

ax.bar(df2.ciudad, df2["precio"], color="C0")

ax2 = ax.twinx()

ax2.plot(df2.ciudad, df2["factorDecoradoDescripcion"], color="C1", marker="D", ms=7)



fig.autofmt_xdate(rotation=75)

ax.set_ylabel("Precio")

ax2.set_ylabel("Coef. Decorado")



ax.tick_params(axis="y", colors="C0")

ax2.tick_params(axis="y", colors="C1")

plt.show();
zonaProp = pd.read_csv('./data/train.csv')

coefDecorado = pd.read_csv('./data/coefdecorado_df.csv')

coefDecorado = coefDecorado.loc[coefDecorado["factorDecoradoDescripcion"] != 0.0]

zonaPropMerge = pd.merge(zonaProp, coefDecorado, on='id', how='inner')



zonaPromedioAntiguedad = zonaPropMerge.groupby(["tipodepropiedad", "provincia"]).agg({"antiguedad":"mean"}).reset_index()

zonaPropMerge["antiguedad"] = zonaPropMerge["antiguedad"].fillna(zonaPromedioAntiguedad["antiguedad"])

promedioAntiguedad = zonaPromedioAntiguedad["antiguedad"].mean()

zonaPropMerge["antiguedad"] = zonaPropMerge["antiguedad"].fillna(promedioAntiguedad)



zonaPropMerge1 = zonaPropMerge.groupby('antiguedad').agg({'factorDecoradoDescripcion': 'mean'}).reset_index()

zonaPropMerge1 = zonaPropMerge1.loc[zonaPropMerge1["factorDecoradoDescripcion"] > 0.10]

zonaPropMerge1 = zonaPropMerge1.loc[zonaPropMerge1["factorDecoradoDescripcion"] < 0.260]

sns.lmplot(x='antiguedad', y='factorDecoradoDescripcion', data=zonaPropMerge1, fit_reg=True, height=7);