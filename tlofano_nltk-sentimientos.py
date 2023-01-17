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
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

nltk.download('punkt')

nltk.download('averaged_perceptron_tagger')

nltk.download('tagsets')
# Ejemplo



sid = SentimentIntensityAnalyzer()

sid.polarity_scores("Excellent.")
import requests

import json



from nltk.sentiment.vader import SentimentIntensityAnalyzer



def aplicar_sentimiento(sentence, contador):

    contador[0] += 1

    print(contador[0])

    api_url = "http://mymemory.translated.net/api/get?q={}&langpair={}|{}".format(sentence, 'es', 'en')

    hdrs ={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',

        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',

        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',

        'Accept-Encoding': 'none',

        'Accept-Language': 'en-US,en;q=0.8',

        'Connection': 'keep-alive'}

    response = requests.get(api_url, headers=hdrs)

    response_json = json.loads(response.text)

    translation = response_json["responseData"]["translatedText"]

    translator_name = "MemoryNet Translation Service"



    sia = SentimentIntensityAnalyzer()

    sentimiento = sia.polarity_scores(translation)

    return sentimiento
def prevent_error_aplicar_sentimiento(descripcion, contador):

    try:

        return aplicar_sentimiento(descripcion, contador)

    except:

        return { 'pos':0, 'neg':0, 'neu':0 }
segmento_random = zonaProp[~zonaProp['descripcion'].isnull()].sample(10000)

contador = [0]

segmento_random['sentimiento'] = segmento_random['descripcion'].apply(lambda descripcion: prevent_error_aplicar_sentimiento(descripcion, contador))
segmento_random['sentimiento_positivo'] = segmento_random['sentimiento'].apply(lambda sentimiento: sentimiento['pos'])

segmento_random['sentimiento_negativo'] = segmento_random['sentimiento'].apply(lambda sentimiento: sentimiento['neg'])

segmento_random['sentimiento_neutro'] = segmento_random['sentimiento'].apply(lambda sentimiento: sentimiento['neu'])
output_file = 'sentimientos_random.csv'

output_dir = Path('./data')

output_dir.mkdir(parents=True, exist_ok=True)

segmento_random[['id', 'sentimiento', 'sentimiento_positivo', 'sentimiento_negativo', 'sentimiento_neutro']].to_csv(output_dir/output_file)