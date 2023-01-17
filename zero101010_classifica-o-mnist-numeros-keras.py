# importações necessárias

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline

# Numeros de seeds, para melhorar a acurácia recomenda se rodar com 30 seeds

np.random.seed(30)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.layers import Conv2D, MaxPooling2D





sns.set(style='white', context='notebook', palette='deep')


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# CARREGANDO DADOS

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
# definir y do treino

y_train = train['label']

X_train = train.drop(labels=["label"],axis=1)



del train

# gráfico de dados resultados 

grafico = sns.countplot(y_train)

# quantidade de cada resultado

y_train.value_counts()

#print(grafico)
# Verificando se possui algum valor null nos dados de treino, para não ter imagens quebradas 

X_train.isnull().any().describe()
# Verificando se possui algum valor null nos dados de treino

test.isnull().any().describe()
# Normalizando os dados 

X_train = X_train/255.0

test = test/255.0
# remodelar imagens para 3 dimensões

X_train = X_train.values.reshape(-1,28,28,1)

X_test = test.values.reshape(-1,28,28,1)

#Separar os dados em 10 classes especificas para a posição que aparece o resultado

Y_train = to_categorical(y_train,num_classes=10)

random_seed = 2

# separar os dados de treino em 2 partes e tirando somente 10% para uma validação

X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size = 0.1,random_state = random_seed)



grafico = plt.imshow(X_train[0][:,:,0])

grafico
# visualizar randomicamente algumas imagens

for i in range(0,6):

    random_num = np.random.randint(0,len(X_train))

    img = X_train[random_num]

    plt.subplot(3,2,i+1)

    plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap('gray'))

# dar um zoom na imagem

plt.subplots_adjust(top=1.4)

plt.show()
#Executa o processo de CNN demonstrado acima, moldando o modelo

model = Sequential()

model.add(Conv2D(32, (5,5), activation='relu', padding='same', input_shape=(28, 28,1)))

model.add(Conv2D(64, (5,5), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



optimizer = Adam()

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



batch_size = 32

epochs = 30



#treionando modelo com CNN montado anteriormente

history = model.fit(X_train,Y_train,batch_size=batch_size,epochs = epochs, validation_split = 0.2, verbose=1,callbacks=[learning_rate_reduction])
#pegando dados de acurácias

history_dict = history.history

#Pegando resultados da acurácia do conjunto de validação

acc = history_dict['accuracy']

print("Acurácias conjunto validação por época: {}".format(acc))

#Pegando o resultados da acurácia do conjunto de treino

val_acc = history_dict['val_accuracy']

print("Acurácias conjunto de treino por época: {}".format(val_acc))

#plotando gráfico para comparar acurácias

range_epochs = range(1,len(acc)+1)

accuracy_val = plt.plot(range_epochs, val_acc, label='Acurácia no conjunto de validação')

accuracy_train = plt.plot(range_epochs, acc, label='Acurácia no conjunto de treino', color="r")

plt.setp(accuracy_val, linewidth=2.0)

plt.setp(accuracy_train, linewidth=2.0)

plt.xlabel('Épocas') 

plt.ylabel('Acurácia')

plt.legend(loc="lower right")

plt.show()
### Aplicando predições depois do treino

predictions = model.predict_classes(X_test)



plt.figure(figsize=(7,14))

for i in range(0,8):

    random_num = np.random.randint(0,len(X_test))

    img = X_test[random_num]

    plt.subplot(6,4,i+1)

    plt.margins(x = 20, y = 20)

    plt.title('Predição: ' + str(predictions[random_num]))

    plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap('gray'))

plt.show()

# Submição 

submission = pd.DataFrame({'ImageID' : pd.Series(range(1,28001)), 'Label' : predictions})

submission.to_csv("submission.csv",index=False)