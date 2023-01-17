#librerias

import os 

import nltk

from nltk.util import bigrams

from nltk.util import ngrams

from nltk.util import everygrams

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

import pandas as pd 
datos = pd.read_csv("../input/grammar-and-online-product-reviews/GrammarandProductReviews.csv", encoding="utf8")

#lower

#grammar = grammar.lower()

#todas las variables

grammar = datos[["name", "manufacturer","reviews.rating" , "reviews.text", "reviews.username"]]



reviews = grammar.loc[:, "reviews.text"]



reviews = reviews.tolist()

#lower case 

reviews = ','.join(str(e) for e in reviews)

reviews = reviews.lower()

#reviews

tokenizer = RegexpTokenizer(r'\w+')

reviews_tokens=tokenizer.tokenize(reviews)

reviews_tokens
#remover stop words

stop_words = set(stopwords.words('english'))

filtered = [w for w in reviews_tokens if not w in stop_words]

#frecuencia

freq_reviews = nltk.FreqDist(filtered)



freq_reviews.most_common(10)
#bigrams

bi_reviews = list(bigrams(filtered))

bi_reviews
#ngrams

tri_reviews = list(ngrams(filtered, n=3))

tri_reviews
#wordcloud

from wordcloud import WordCloud

cloud = list(freq_reviews.most_common(10))

cloud

wordcloud = WordCloud().generate("great product movie review part promotion collected love use good")

import matplotlib.pyplot as plt

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")
# datos para el modelo

lineList1 = [line.rstrip('\n') for line in open("../input/sentimentanalysis/amazon_cells_labelled.txt")]

lineList2 = [line.rstrip('\n') for line in open("../input/sentimentanalysis/imdb_labelled.txt")]

lineList3 = [line.rstrip('\n') for line in open("../input/sentimentanalysis/yelp_labelled.txt")]



lineList = lineList1 +lineList2 +lineList3

tokenizer = RegexpTokenizer(r'\w+')

limpio = []

for line in lineList:

    line = line.lower()

    line_tokens=tokenizer.tokenize(line)

    limpio.append(line_tokens)



def form_sent(sent):

    return {word: True for word in nltk.word_tokenize(sent)}



train_data = []

for element in limpio:

    listTemp = []

    strtemp = ""

    for value in range(0,len(element)-1):

        strtemp = strtemp + element[value] + " "

    cadena = form_sent(strtemp)

    listTemp.append(cadena)

    if(element[len(element)-1] == '0'):

        listTemp.append('neg')

    elif(element[len(element)-1] == '1'):

        listTemp.append('pos')

    train_data.append(listTemp)



print(train_data)
# Entrenamiento del modelo 

from nltk.classify import NaiveBayesClassifier

model = NaiveBayesClassifier.train(train_data)

model.classify(form_sent('This is a good article'))
def modelres(frase):

    #print(frase, '\n',model.classify(form_sent(frase)) )

    return model.classify(form_sent(str(frase)))



grammar['reviews.meaning'] = grammar['reviews.text'].apply(modelres)
# Tabla de contingencia class / survived

productos = pd.crosstab(index=grammar['name'],

            columns=grammar['reviews.meaning'], margins=True)



#productos con menor calidad

productos.sort_values(by='pos',ascending=0).head(10)
#productos con peor calidad

productos.sort_values(by='neg',ascending=0).head(10)
# Tabla de contingencia class / survived

user_rev = datos.groupby(datos['reviews.username']).size().reset_index(name='Count').rename(columns={'reviews.username':'users'})

user_rev.sort_values(by='Count',ascending=0).head(10)
# Tabla de contingencia usuarios - reviews.meaning

user_rev_mean = pd.crosstab(index=grammar['reviews.username'],

            columns=grammar['reviews.meaning'], margins=True).apply(lambda r: r/len(datos) *100,

                                axis=1)
#usuarios que más reviews negativos dan en promedio.

user_rev_mean.sort_values(by='neg',ascending=0).head(10)
#usuarios que más reviews positivos dan en promedio.

user_rev_mean.sort_values(by='pos',ascending=0).head(10)
# Tabla de contingencia productores - review.meaning

productores = pd.crosstab(index=grammar['manufacturer'],

            columns=grammar['reviews.meaning'], margins=True)                       

# Cuáles son los productores que tienen productos de mejor calidad.

productores.sort_values(by='pos',ascending=0).head(10)
#Cuáles son los productores que tienen productos de peor calidad.

productores.sort_values(by='neg',ascending=0).head(10)