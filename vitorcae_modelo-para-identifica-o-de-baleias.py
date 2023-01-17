import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch
from torchvision import transforms


import tensorflow as tf
import random
import time
import cv2

from skimage import io #Pacote para ler e gravar imagens em vários formatos.
from pylab import rcParams 

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import data, color

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D,BatchNormalization,AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Conv2D, MaxPool2D
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import (
    random_rotation, random_shift, random_shear, random_zoom,
    random_channel_shift,img_to_array, ImageDataGenerator)

import numpy as np
import pandas as pd

import warnings
from glob import glob

%matplotlib inline

img_treino= "../input/humpback-whale-identification/train/"
#img_test= "../input/test-foto/"
base_treino="../input/humpback-whale-identification/train.csv"
IMG_SIZE = 64
#carrega a base em csv de treino
df_treino = pd.read_csv(base_treino) 

# encontra os elementos distintos de uma matriz
id_dist = np.unique(df_treino[['Id']].values) 

#Elaboração da variável classe_dist 
classe_dist = {}

#Elaboração da variável classe_id_dist 
classe_id_dist = {}

#Preenchimento das variaveis criando um dicionário na classe_dist e classe_id_dist
for i in range(len(id_dist)):
    classe_dist[id_dist[i]] = i
    classe_id_dist[i] = id_dist[i]

# Adiciona um nova coluna chamada class_id na tabela  df_treino  contendo os id criados
df_treino['classes_id'] = df_treino.apply (lambda row: classe_dist.get(row['Id']),axis=1)
df_treino.head(15)
# Criação da Função para plotar as imagens
def show_img(image):
    plt.imshow(image)
#Criação da Função que plota as imagens carregaas em show_img
def plot_img(images):
    
    #Define o tamanho da imagem
    #rcParams['figure.figsize'] = 13, 8
    rcParams['figure.figsize'] = 14, 8
    
    #Insere o Mapa de Cores
    #plt.jet()
    plt.gray()
    
    fig = plt.figure()
    
    #Loop elaborado para retornar a menor valor em 9 iterações
    for i in range(min(9, images.shape[0])):
        fig.add_subplot(3, 3, i+1)
        show_img(images[i])
    plt.show()   
#Função para redimencionar o tamanho das imagens
def LoadImage(img_path):
    
    #Carregar uma imagem do arquivo e cria uma escala de cinza 
    image = color.rgb2gray(io.imread(img_path))
    
    #Redimencionar o tamanho das imagens
    image_resized = resize(image,(IMG_SIZE,IMG_SIZE))
    return image_resized[:,:] / 255.

#Função para carregar os  dados de imagens e identificação de classes
def LoadImageData(path):
    xs = []
    ys = []
    #for ex_paths in paths:
    for index, row in df_treino.iterrows(): 
        
        #Armazena o caminha de cada uma das imagens
        img_path = path+row['Image']
        
        #Carega a imagem que tem seu caminho armazenado na variável acima
        igm = LoadImage(img_path)
        
        #Add os  elementos na lista xs
        xs.append(igm)
        
        #Add os elementos na lista ys
        ys.append(row['classes_id'])
        
    return np.array(xs),np.array(ys)
#Carrega em X_train as imagens contidas em xs e a classe delas em Y_train contidas em ys
X_train,Y_train = LoadImageData(img_treino)
print("Base Carregada")
#Verificação do  número de elementos em cada dimensão
print("X_train ",X_train.shape)
print("Y_train ",Y_train.shape)

#Verificação  aleatótia das imagens
xs = [random.randint(0, X_train.shape[0]-1) for _ in range(9)]   
print("XS ",xs)
plot_img(X_train[xs])


X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


#Mudança da Classe_dist dos id para 0 e 1 em seus formatos 
Y_train = keras.utils.to_categorical(Y_train,num_classes=len(classe_dist))

print(np.shape(X_train))
print(np.shape(Y_train))

#Criação da função para rodar o modelo CNN
def cnn():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), strides = (1, 1), input_shape = (IMG_SIZE, IMG_SIZE, 1)))
    model.add(BatchNormalization(axis = 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), strides = (1,1)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3)))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.6))
    model.add(Dense(5005, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model
#Primeiro modelo com 100 Epocas e batch_size de aproximadamente 20% do tamanho da base
model1 = cnn()
history1 = model1.fit(X_train, Y_train, epochs=100, batch_size=500, verbose=1)
#Segun  modelo com 100 Epocas e batch_size de aproximadamente 10% do tamanho da base
model2 = cnn()
history2 = model2.fit(X_train, Y_train, epochs=100, batch_size=250, verbose=1)
#Terceiro  modelo com 100 Epocas e batch_size de aproximadamente 5% do tamanho da base
model3 = cnn()
history3 = model3.fit(X_train, Y_train, epochs=100, batch_size=127, verbose=1)
#Sexto  modelo com 100 Epocas e batch_size de aproximadamente 2,5% do tamanho da base
model6 = cnn()
history6 = model6.fit(X_train, Y_train, epochs=100, batch_size=60, verbose=1)
#Quarto  modelo com 100 Epocas e batch_size de aproximadamente 1,62% do tamanho da base
model4 = cnn()
history4 = model4.fit(X_train, Y_train, epochs=100, batch_size=30, verbose=1)
#Quinto  modelo com 100 Epocas e batch_size de aproximadamente 0,5% do tamanho da base
model5 = cnn()
history5 = model5.fit(X_train, Y_train, epochs=100, batch_size=10, verbose=1)
#Decimo modelo com 200 Epocas e batch_size de aproximadamente 0,25% do tamanho da base
model61 = cnn()
history61 = model61.fit(X_train, Y_train, epochs=100, batch_size=5, verbose=1)
#Setimo  modelo com 200 Epocas e batch_size de aproximadamente 5% do tamanho da base
model7 = cnn()
history7 = model7.fit(X_train, Y_train, epochs=200, batch_size=127, verbose=1)
#Oitavo  modelo com 200 Epocas e batch_size de aproximadamente 2,5% do tamanho da base
model8 = cnn()
history8 = model8.fit(X_train, Y_train, epochs=200, batch_size=60, verbose=1)
#Nono  modelo com 200 Epocas e batch_size de aproximadamente 0,31% do tamanho da base
model9 = cnn()
history9 = model9.fit(X_train, Y_train, epochs=200, batch_size=30, verbose=1)
#Decimo modelo com 200 Epocas e batch_size de aproximadamente 0,31% do tamanho da base
model10 = cnn()
history10 = model10.fit(X_train, Y_train, epochs=200, batch_size=250, verbose=1)
#Decimo modelo com 200 Epocas e batch_size de aproximadamente 0,31% do tamanho da base
model101 = cnn()
history101 = model101.fit(X_train, Y_train, epochs=200, batch_size=10, verbose=1)
plt.plot(history1.history['acc'], color='blue', linewidth = 2, label="batch_size=500") 
plt.plot(history2.history['acc'], color='red', linewidth = 2, label="batch_size=250") 
plt.plot(history3.history['acc'], color='green', linewidth = 2, label="batch_size=125") 
plt.plot(history6.history['acc'], color='black', linewidth = 2, label="batch_size=60")
plt.plot(history4.history['acc'], color='orange', linewidth = 2, label="batch_size=30") 
plt.plot(history61.history['acc'], color='gray', linewidth = 2, label="batch_size=10")
plt.plot(history5.history['acc'], color='yellow', linewidth = 2, label="batch_size=5")
plt.rcParams['figure.figsize']=(10,10)
plt.title('Acurácia do Modelo CNN - 100 Épocas')
plt.ylabel('Acurácia')
plt.xlabel('Épocas')
plt.legend()
plt.show()
plt.plot(history7.history['acc'], color='blue', linewidth = 2, label="batch_size=250") 
plt.plot(history8.history['acc'], color='red', linewidth = 2, label="batch_size=125") 
plt.plot(history9.history['acc'], color='green', linewidth = 2, label="batch_size=60")
plt.plot(history10.history['acc'], color='yellow', linewidth = 2, label="batch_size=30")
plt.plot(history101.history['acc'], color='black', linewidth = 2, label="batch_size=10")
plt.rcParams['figure.figsize']=(10,10)
plt.title('Acurácia do Modelo CNN - 200 Épocas')
plt.ylabel('Acurácia')
plt.xlabel('Épocas')
plt.legend()
plt.show()
plt.plot(history1.history['loss'], color='blue', linewidth = 2, label="batch_size=500") 
plt.plot(history2.history['loss'], color='red', linewidth = 2, label="batch_size=250") 
plt.plot(history3.history['loss'], color='green', linewidth = 2, label="batch_size=125") 
plt.plot(history6.history['loss'], color='black', linewidth = 2, label="batch_size=60")
plt.plot(history4.history['loss'], color='orange', linewidth = 2, label="batch_size=30") 
plt.plot(history5.history['loss'], color='yellow', linewidth = 2, label="batch_size=10")
plt.plot(history61.history['loss'], color='gray', linewidth = 2, label="batch_size=5")
plt.rcParams['figure.figsize']=(10,10)
plt.legend()
plt.title('Perda do Modelo CNN - 100 Épocas')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.xlabel('Épocas')
plt.show()
plt.plot(history7.history['loss'], color='blue', linewidth = 2, label="batch_size=250") 
plt.plot(history8.history['loss'], color='red', linewidth = 2, label="batch_size=125") 
plt.plot(history9.history['loss'], color='green', linewidth = 2, label="batch_size=60") 
plt.plot(history10.history['loss'], color='yellow', linewidth = 2, label="batch_size=30") 
plt.plot(history101.history['loss'], color='yellow', linewidth = 2, label="batch_size=10")
plt.rcParams['figure.figsize']=(10,10)
plt.legend()
plt.title('Perda do Modelo CNN - 200 Épocas')
plt.ylabel('Loss')
plt.xlabel('Épocas')
plt.xlabel('Épocas')
plt.show()
opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
#Criação da função para rodar o modelo CNN
def cnn1():
    modelo = Sequential()
    modelo.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (IMG_SIZE, IMG_SIZE, 1)))
    modelo.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
    modelo.add(MaxPool2D(pool_size = (2,2)))
    modelo.add(Dropout(0.25))
    modelo.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
    modelo.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
    modelo.add(MaxPool2D(pool_size = (2,2), strides=(2,2)))
    modelo.add(Dropout(0.25))
    modelo.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
    modelo.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
    modelo.add(MaxPool2D(pool_size = (2,2), strides=(2,2)))
    modelo.add(Dropout(0.25))
    modelo.add(Flatten())
    modelo.add(Dense(256, activation = 'relu'))
    modelo.add(BatchNormalization())
    modelo.add(Dense(Y_train.shape[1], activation = "softmax"))
    modelo.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    modelo.summary()
    return modelo   

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
#Decimo modelo com 200 Epocas e batch_size de aproximadamente 10% do tamanho da base
modelo11 = cnn1()
history11 = modelo11.fit(X_train, Y_train, epochs=100, batch_size=250, verbose=1)
#Decimo modelo com 200 Epocas e batch_size de aproximadamente 5% do tamanho da base
modelo12 = cnn1()
history12 = modelo12.fit(X_train, Y_train, epochs=100, batch_size=125, verbose=1)
#Decimo modelo com 200 Epocas e batch_size de aproximadamente 2,5% do tamanho da base
modelo13 = cnn1()
history13 = modelo13.fit(X_train, Y_train, epochs=100, batch_size=60, verbose=1)
#Decimo modelo com 200 Epocas e batch_size de aproximadamente 1,62% do tamanho da base
modelo14 = cnn1()
history14 = modelo14.fit(X_train, Y_train, epochs=100, batch_size=30, verbose=1)
#Decimo modelo com 200 Epocas e batch_size de aproximadamente 0,5% do tamanho da base
modelo15 = cnn1()
history15 = modelo15.fit(X_train, Y_train, epochs=100, batch_size=10, verbose=1)
#Decimo modelo com 200 Epocas e batch_size de aproximadamente 0,5% do tamanho da base
modelo16 = cnn1()
history16 = modelo16.fit(X_train, Y_train, epochs=100, batch_size=500, verbose=1)
plt.plot(history16.history['acc'], color='gray', linewidth = 2, label="batch_size=500")
plt.plot(history11.history['acc'], color='blue', linewidth = 2, label="batch_size=250") 
plt.plot(history12.history['acc'], color='red', linewidth = 2, label="batch_size=125") 
plt.plot(history13.history['acc'], color='green', linewidth = 2, label="batch_size=60")
plt.plot(history14.history['acc'], color='yellow', linewidth = 2, label="batch_size=30")
plt.plot(history15.history['acc'], color='black', linewidth = 2, label="batch_size=10")
plt.title('Acurácia do Modelo CNN - 100 Épocas e 16 camadas')
plt.legend()
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.xlabel('Épocas')
plt.show()
plt.plot(history16.history['loss'], color='blue', linewidth = 2, label="batch_size=500") 
plt.plot(history11.history['loss'], color='blue', linewidth = 2, label="batch_size=250") 
plt.plot(history12.history['loss'], color='red', linewidth = 2, label="batch_size=125") 
plt.plot(history13.history['loss'], color='green', linewidth = 2, label="batch_size=60") 
plt.plot(history14.history['loss'], color='yellow', linewidth = 2, label="batch_size=30") 
plt.plot(history15.history['loss'], color='black', linewidth = 2, label="batch_size=10") 
plt.title('Perda do Modelo CNN - 100 Épocas - 16 Camadas')
plt.legend()
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.xlabel('Épocas')
plt.show()
plt.plot(history4.history['acc'], color='orange', linewidth = 2, label="batch_size - 1,62% -  11 L") 
plt.plot(history8.history['acc'], color='red', linewidth = 2, label="batch_size - 2,5% - 11 L") 
plt.plot(history13.history['acc'], color='green', linewidth = 2, label="batch_size -2,5% - 14 L")
plt.legend()
plt.title('Acurácia dos melhores Modelos')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.xlabel('Épocas')
plt.show()