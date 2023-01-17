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
descripciones = zonaProp['descripcion']
# Corrección del encoding, y filtrado de elementos inncesarios en los textos



def palabras_totales(texto, array):

    array.extend(sanitize(texto))





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
# Retorna todas las palabras separadas por un espacio (en un string)



seperator = ' '

palabras = []

zonaProp['descripcion'].apply(lambda descripcion: palabras_totales(descripcion, palabras))

palabras_str = seperator.join(palabras)
# TAGGER SIN ENTREDAR



# Taggea todas las palabras(recibe un arrar de cada una de las palabras que hay, las repeticiones 

# aparecen tantas veces se haya repetido la palabra)



total_palabras_tag = nltk.pos_tag(palabras)
# Esto es el WordCloud de todas las palabras, sin importar clasificacion

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_str)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son dolar: `$ -$ --$ A$ C$ HK$ M$ NZ$ S$ U.S.$ US$`

palabras_dolar = filtrado_nltk(total_palabras_tag, '$')

seperator = ' '

palabras_dolar = seperator.join(palabras_dolar)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_dolar)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son conjunciones

palabras_conjunciones = filtrado_nltk(total_palabras_tag, 'CC')

seperator = ' '

palabras_conjunciones = seperator.join(palabras_conjunciones)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_conjunciones)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son numericas o cardinales

palabras_cardinal = filtrado_nltk(total_palabras_tag, 'CD')

seperator = ' '

palabras_cardinal = seperator.join(palabras_cardinal)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_cardinal)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son determiner

palabras_chan = filtrado_nltk(total_palabras_tag, 'DT')

seperator = ' '

palabras_chan = seperator.join(palabras_chan)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_chan)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son palabras extrangeras

palabras_extranjeras = filtrado_nltk(total_palabras_tag, 'FW')

seperator = ' '

palabras_extranjeras = seperator.join(palabras_extranjeras)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_extranjeras)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son palabras prepociciones

palabras_prepo = filtrado_nltk(total_palabras_tag, 'IN')

seperator = ' '

palabras_prepo = seperator.join(palabras_prepo)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_prepo)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son palabras adjetivos numericos

palabras_adjetivo_numerico = filtrado_nltk(total_palabras_tag, 'JJ')

seperator = ' '

palabras_adjetivo_numerico = seperator.join(palabras_adjetivo_numerico)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_adjetivo_numerico)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son palabras comparativoas

palabras_comparativas = filtrado_nltk(total_palabras_tag, 'JJR')

seperator = ' '

palabras_comparativas = seperator.join(palabras_comparativas)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_comparativas)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son palabras superlativas

palabras_superlativas = filtrado_nltk(total_palabras_tag, 'JJS')

seperator = ' '

palabras_superlativas = seperator.join(palabras_superlativas)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_superlativas)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son palabras sstantivo

palabras_sustantivo = filtrado_nltk(total_palabras_tag, 'NN')

seperator = ' '

palabras_sustantivo = seperator.join(palabras_sustantivo)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_sustantivo)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son palabras sstantivo

palabras_sustantivo = filtrado_nltk(total_palabras_tag, 'NN')

seperator = ' '

palabras_sustantivo = seperator.join(palabras_sustantivo)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_sustantivo)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son palabras adverbio

palabras_adverbio = filtrado_nltk(total_palabras_tag, 'RB')

seperator = ' '

palabras_adverbio = seperator.join(palabras_adverbio)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_adverbio)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son palabras

palabras_chin = filtrado_nltk(total_palabras_tag, 'UH')

seperator = ' '

palabras_chin = seperator.join(palabras_chin)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_chin)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son palabras verbos

palabras_verbos = filtrado_nltk(total_palabras_tag, 'VB')

seperator = ' '

palabras_verbos = seperator.join(palabras_verbos)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_verbos)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# FAQ https://nlp.stanford.edu/software/spanish-faq.html

# Link descarga: https://nlp.stanford.edu/software/stanford-postagger-full-2018-10-16.zip



from nltk.tag import StanfordPOSTagger

jar = './tagger-standford/stanford-postagger-3.9.2.jar'

model = './tagger-standford/spanish.tagger'



import os

# java_path = "/usr/lib/jvm/java-8-oracle/"

os.environ['JAVAHOME'] = os.environ['JAVA_HOME']



pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')



# palabras

total_palabras_tag_ES = pos_tagger.tag(palabras)
total_palabras_tag_ES
# Filtrado de palabras que son palabras sustantivos

palabras_sustantivos = filtrado_nltk_arr(total_palabras_tag_ES, ['nc00000', 'nc0n000', 'nc0p000', 'nc0s000', 'np00000'])

seperator = ' '

palabras_sustantivos = seperator.join(palabras_sustantivos)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_sustantivos)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son palabras Adjetivos

palabras_adjetivos = filtrado_nltk_arr(total_palabras_tag_ES, ['ao0000', 'aq0000'])

seperator = ' '

palabras_adjetivos = seperator.join(palabras_adjetivos)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_adjetivos)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Filtrado de palabras que son palabras Verbos



palabras_verbos = filtrado_nltk_arr(total_palabras_tag_ES, ['va00000', 'vag0000', 'vaic000', 'vaif000', 'vaii000', 'vaip000', 'vais000', 'vam0000', 'van0000', 'vap0000', 'vasi000', 'vasp000', 'vmg0000', 'vmic000', 'vmif000', 'vmii000', 'vmip000', 'vmis000', 'vmm0000', 'vmn0000', 'vmp0000', 'vmsi000', 'vmsp000', 'vsg0000', 'vsic000', 'vsif000', 'vsii000', 'vsip000', 'vsis000', 'vsm0000', 'vsn0000', 'vsp0000', 'vssf000', 'vssi000', 'vssp000'])

seperator = ' '

palabras_verbos = seperator.join(palabras_verbos)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_verbos)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Nube de palabras adjetivos + Sustantivos



palabras_adjetivos_sustantivos = palabras_adjetivos.extend(palabras_sustantivos)

seperator = ' '

palabras_adjetivos_sustantivos = seperator.join(palabras_adjetivos_sustantivos)



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_adjetivos_sustantivos)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
# Retorna todas las palabras separadas por un espacio (en un string)



palabras_titulo = []

seperator = ' '

zonaProp['titulo'].apply(lambda titulo: palabras_totales(titulo, palabras_titulo))

palabras_str_titulos = seperator.join(palabras_titulo)
# Esto es el WordCloud de todas las palabras, sin importar clasificacion

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(palabras_str_titulos)



fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()