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

df_props_full = pd.read_csv(path + "train.csv")
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

df_props_full['precio'] = df_props_full['precio'].astype(pd.UInt32Dtype())
df_props_full['year'] = df_props_full['fecha'].dt.year

df_props_full['month'] = df_props_full['fecha'].dt.month
df_props_full['first_fortnight'] = df_props_full['fecha'].apply(lambda fecha: 1 if fecha.day < 15 else 0)
df_props_full.groupby('first_fortnight').agg('size').to_frame()
df_dollar = pd.read_csv(path + 'dollar.csv')

print(df_dollar.dtypes)

print('\n')

print(df_dollar.shape)

df_dollar.head(2)
# Con describe identifico si hay valores nulos

df_dollar.describe()
# Muestro las lineas con valores nulos

df_dollar[df_dollar.isna().any(axis=1)]
# Analiso los últimos registros

df_dollar.loc[1340:1343]
df_dollar = df_dollar.dropna()

df_dollar.describe()
df_dollar['Cierre'] = pd.to_numeric(df_dollar['Cierre'])

df_dollar['Cierre'] = df_dollar['Cierre'].round(3)

df_dollar['Fecha'] = pd.to_datetime(df_dollar['Fecha'], format='%d.%m.%Y')

df_dollar = df_dollar.set_index('Fecha')

df_dollar = df_dollar.loc[:, 'Cierre'].to_frame()
# Rango de fechas

print(df_dollar.index.min())

print(df_dollar.index.max())
# Agrego fechas faltantes (Sabados y Domingos) con valor 0

idx = pd.date_range(start='2011-12-12', end='2017-01-31')

df_dollar = df_dollar.reindex(idx, fill_value=0)
df_dollar.head(8)
# Cuando se trata de una fecha que corresponde a un Sabado o Domingo no se tiene infromación sobre Cierre

# Le asigno el valor correspondiente al Viernes previo

for i in range(0, len(df_dollar)):

    if (df_dollar.iloc[i]['Cierre'] == 0):

        df_dollar.iloc[i]['Cierre'] = df_dollar.iloc[i-1]['Cierre']
df_dollar.head(8)
price_dates = df_props_full.loc[:,['fecha','precio','year','month','first_fortnight']]

price_dates['fecha'] = price_dates['fecha'].apply(lambda x: x.replace(hour=0, minute=0, second=0)) # Seteo tiempo a 00:00:00 para join

price_dates = price_dates.set_index('fecha')

price_dates = price_dates.join(df_dollar, how='left')

price_dates = price_dates.reset_index()

price_dates.rename(columns = {'index' : 'fecha'}, inplace=True)

price_dates.describe()
price_dates.rename(columns = {'Cierre' : 'MEX_to_USD', 'precio' : 'Precio_MEX'}, inplace=True)

price_dates.head()
price_dates['Precio_USD'] = price_dates['Precio_MEX'] * price_dates['MEX_to_USD']

price_dates['Precio_USD'] = price_dates['Precio_USD'].astype(int)

price_dates.head()
words_titulo = df_props_full['titulo'].to_frame()

words_titulo = words_titulo['titulo'].str.lower().to_frame()
words_titulo = words_titulo.dropna()

words_titulo = words_titulo.groupby('titulo').agg({'titulo':'count'})

words_titulo.index.names = ['index']
words_titulo.rename(columns = {'titulo' : 'appearences'}, inplace=True)

words_titulo = words_titulo.reset_index()

words_titulo.rename(columns = {'index' : 'titulo'}, inplace=True)
import nltk

nltk.download('punkt')
words_titulo['token'] = words_titulo['titulo'].apply(nltk.word_tokenize)

words_titulo.head()
# Retorna una lista de los trigramas generados a partir de la serie

def trigrams_list(x):

    return list(nltk.ngrams(x,3))
words_titulo['trigrams'] = words_titulo['token'].apply(trigrams_list)

words_titulo.head(20)
trigrams = pd.Series([trigram for title_trigrams in words_titulo['trigrams'] for trigram in title_trigrams])

trigrams = trigrams.rename('trigram')

trigrams = trigrams.to_frame()

trigrams.head()
trigrams = trigrams.groupby('trigram').agg('size')



trigrams = trigrams.to_frame()



trigrams.columns = ['appearences']

trigrams = trigrams.sort_values('appearences',ascending=False)
trigrams = trigrams.reset_index()
trigrams['trigram'] = trigrams['trigram'].str.join(",")

trigrams['trigram'] = trigrams['trigram'].str.replace(r'\,',' ')
trigrams.head(20)
# Retorna una lista de las palabras en la serie

def words_list(x):

    return list(nltk.ngrams(x,1))
words_titulo['words'] = words_titulo['token'].apply(words_list)

words = pd.Series([word for title_words in words_titulo['words'] for word in title_words])

words = words.rename('words')

words = words.to_frame()
words = words.groupby('words').agg('size')

words = words.to_frame()

words.columns = ['appearences']

words = words.sort_values('appearences',ascending=False)
words = words.reset_index()

words['words'] = words['words'].str.join(",")
# Conservo palabras con 4 o  mas caracteres

words = words.loc[words['words'].apply(lambda x: len(x) > 3)]
# Creo diccionario: Palabra:Frequencia

words_dicc = dict(zip(words['words'], words['appearences']))
# Cantidad de palabras

len(words_dicc)
from wordcloud import WordCloud
# Utilizo los 150 terminos con mayor frecuencia

wordcloud = WordCloud(background_color="white", width=1000, height=1000, colormap='cividis', max_words=150).generate_from_frequencies(frequencies=words_dicc)

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
words_desc = df_props_full['descripcion'].to_frame()

words_desc = words_desc['descripcion'].str.lower().to_frame()
words_desc = words_desc.dropna()

words_desc = words_desc.groupby('descripcion').agg({'descripcion':'count'})

words_desc.index.names = ['index']
words_desc.rename(columns = {'descripcion' : 'appearences'}, inplace=True)

words_desc = words_desc.reset_index()

words_desc.rename(columns = {'index' : 'descripcion'}, inplace=True)
words_desc.head()
#Utilizo una expresion regular para borrar el texto dentro de tags html

import re



def cleanhtml(descripcion_html):

    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

    descripcion_txt = re.sub(cleanr, '', descripcion_html)

    return descripcion_txt
words_desc['descripcion'] = words_desc['descripcion'].apply(cleanhtml)
import unidecode
# Elimino acentuacion

words_desc['descripcion'] = words_desc['descripcion'].apply(unidecode.unidecode)
words_desc['token'] = words_desc['descripcion'].apply(nltk.word_tokenize)

words_desc.head()
# Retorna una lista de las palabras en la serie

def words_list(x):

    return list(nltk.ngrams(x,1))
words_desc['words'] = words_desc['token'].apply(words_list)

words_desc_apps = pd.Series([word for desc_words in words_desc['words'] for word in desc_words])

words_desc_apps = words_desc_apps.rename('words')

words_desc_apps = words_desc_apps.to_frame()
words_desc_apps = words_desc_apps.groupby('words').agg('size')

words_desc_apps = words_desc_apps.to_frame()

words_desc_apps.columns = ['appearences']

words_desc_apps = words_desc_apps.sort_values('appearences',ascending=False)
words_desc_apps = words_desc_apps.reset_index()

words_desc_apps['words'] = words_desc_apps['words'].str.join(",")
# Conservo palabras con 4 o  mas caracteres

words_desc_apps = words_desc_apps.loc[words_desc_apps['words'].apply(lambda x: len(x) > 3)]
# Creo diccionario: Palabra:Frequencia

words_desc_dicc = dict(zip(words_desc_apps['words'], words_desc_apps['appearences']))
#Elimino para que quedo con más repeticiones

words_desc_dicc.pop('para')

#Corrijo 'ñ' eliminada de palabras principales

words_desc_dicc['baño'] = words_desc_dicc.pop('bano')

words_desc_dicc['baños'] = words_desc_dicc.pop('banos')
# Cantidad de palabras

len(words_desc_dicc)
# Utilizo los 150 terminos con mayor frecuencia

wordcloud = WordCloud(background_color="white", width=1000, height=1000, colormap='cividis', max_words=150).generate_from_frequencies(frequencies=words_desc_dicc)

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()