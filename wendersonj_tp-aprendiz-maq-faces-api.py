# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#!pip install pydotplus

#!pip install dtreeviz



import pandas as pd

import numpy as np

from sklearn import datasets, tree

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.feature_extraction import DictVectorizer

from sklearn.preprocessing import LabelEncoder

import asyncio

import io

import glob

import os

import sys

import time

import uuid

import requests

import json

from urllib.parse import urlparse

from io import BytesIO

from PIL import Image, ImageDraw



!pip install --upgrade azure-cognitiveservices-vision-face

from azure.cognitiveservices.vision.face import FaceClient

from msrest.authentication import CognitiveServicesCredentials

from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType



!mkdir -p /kaggle/working/data

!cp  -r /kaggle/input/* /kaggle/working/data  



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#acesso à casa branca

#Laura_Bush + George_W_Bush | Hugo_Chavez + Serena_Williams



#mas a ideia é: (finge que é um diagrama bonitinho)

#criar um campo de autorizado -> classificar as imagens em autorizadas e não-autorizadas (pelo nome) -> fazer a chamada na API para pegar os atributos (Características das faces) das imagens ->  guardar os atributos ao invés da imagem -> criar um arquivo com os registros de cada face contendo atributos e se é autorizada -> dividir base em teste e treino -> f
data = pd.read_csv('../input/lfw-dataset/people.csv')

print(data.describe())

print(data.head(15))



#print(data.head(),'\n')

print('SOMA IMAGENS:',data['images'].sum(),'\n')

print(data['images'].describe(),'\n')



#print([i for i in data['images'] >= 10])

data=data[data['images'] >= 10]  #filtra os registros que possuem no minimo 10 imagens
nomes=pd.unique(data['name'])

nomes = [(str(x)) for x in nomes]



data['caminho'] = ''
caminhos=[]

for dirname, _, filenames in os.walk('/kaggle/working/data/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/'):

    for filename in filenames: 

        caminhos.append(dirname+'/'+filename)



#caminhos=[dirname.join(filenames) for dirname,_, filenames in os.walk('/kaggle/working/data/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/')]



print("Temos {0} imagens".format(len(caminhos)))

print(caminhos[:3])
dataset=pd.DataFrame(columns=["id_nome", "nome", 'caminho'])



#print(nomes)

dataset

lim=10 #limite de 10 imagens por pessoa / 10 caminhos de imagens

i = 0



#geração do arquivo contendo os registros com o caminho de cada imagem, limitadas a número (lim) de registro por nome

#max 100 pessoas -> [:100]

for n in nomes[:10]:

    #print(c)

    for c in caminhos:

        if i < lim:

            if n+'_0' in c: #'_0' pois há primeiros nomes iguais. isso garante que eu pego o nome completo

                dataset = dataset.append({'id_nome':nomes.index(n), 'nome':n, 'caminho':c}, ignore_index = True)

                i = i+1

        else:

            break

    i=0







dataset
#Exporta o arquivo

dataset.to_csv('caminhos.csv', index=True)
KEY = '04a03b54d4e04f0eb7c300274ec12f9b'

ENDPOINT = 'https://face-student.cognitiveservices.azure.com/'

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

face_api_url = ENDPOINT+'face/v1.0/detect'

#headers = {'Ocp-Apim-Subscription-Key': KEY}
train_face_ids = []

faces_detected = []

names = []



params = {

    'returnFaceId': 'true',

    'returnFaceLandmarks': 'true',

    'returnFaceRectangle':'false',

    'returnFaceAttributes': 'age,gender,smile,headPose,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',    

}



for path in dataset['caminho'][:2]:

    #print(path)

    data=open(path, 'r+b').read()

    headers = {'Ocp-Apim-Subscription-Key': KEY, 'Content-Type':'application/octet-stream', 'Content-Length':str(len(data))}

    response = requests.post(face_api_url, params=params, data=data, headers=headers)

    names.append(path.split('/')[7])

    

    

    for face in response.json():

        train_face_ids.append(face['faceId'])

        faces_detected.append(response.json()[0])
#[print(x) for x in faces_detected]

print(faces_detected[0])
def generate_FL(df):

    

    """

    Tranforma o Dataframe FaceLandmarks.

    

    Padrão de entrada 

        

        atributo_1 atributo_2

    x   0.0        0.0

    y   1.0        1.0

    

    Padrão saída

    

    atributo_1_x atributo_1_y atributo_2_x atributo_2_y

     0.0           1.0          0.0         1.0



    Parameters

    ----------

    df : pd.Dataframe

        

    Returns

    -------

    result : pd.Dataframe

    

    Obs.: não são utilizados a marcação de 'faceRetangle' e os atributos 'smile', 'hair' e 'acessories'.

    """

    

    tuples = []

    for key, values in df.items():

        tuples.append((key + "_x", values[0]))

        tuples.append((key + "_y", values[1]))

    result = pd.DataFrame.from_dict(dict(tuples),orient='index').T

    

    return  result

  

def generate_FA(df, columns_accept):

    

    columns_except = ['smile','gender','age','glasses','accessories'] 

    tuples = []

    for column in df.columns: 

       if(column in columns_accept):

        if(column not in columns_except):

             for key, values in df[column].items():

                if(values != 'hair'):

                    for key_2, values_2 in values.items():

                        tuples.append((column + "_" + key_2, values_2))

        else:  

            if(column != 'accessories'):

                tuples.append((column, df[column]))

    return  pd.DataFrame.from_dict(dict(tuples),orient='index')



def generate_new_df(faces_detected):

    

    """

    Realiza o tratamento dos Dataframes e gera um Dataframe único para todas as imagens da base de dados.

    

    Parameters

    ----------

    faces_detected : pd.Dataframe

        

    Returns

    -------

    result : pd.Dataframe

    

    """



    lista_df =[]

    for i in range(len(faces_detected)):

    

        df_FL = pd.DataFrame.from_dict(faces_detected[i]['faceLandmarks'])

        df_FA = pd.DataFrame.from_dict(faces_detected[i]['faceAttributes'], orient='index').T



        df_FL = generate_FL(df_FL)

        df_FA = generate_FA(df_FA, ['gender','age'])



        df = pd.concat([df_FA.T, df_FL], axis=1)



        lista_df.append(df)

    

    result = pd.concat(lista_df)

    

    return result
features = generate_new_df(faces_detected)

labels = pd.DataFrame({'labels': names})  #acho q nao precisa de outro dataframe para labels

features.insert(loc=0, column='label', value=names, allow_duplicates=True)
print(features.head(2))
dataset.to_csv('atributos.csv', index=True) #Exporta a base da dos completa (atributos + labels)