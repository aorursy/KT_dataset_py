# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
todosDatos = pd.read_csv("../input/inteligencia-sanitaria-curso-introduccin-a-python/COVID19MEXICOsmall.csv")
todosDatos
#######################################################
############# PREPARACIÓN DE DATOS ####################
#######################################################
# Leer archivo
todosDatos = pd.read_csv("../input/inteligencia-sanitaria-curso-introduccin-a-python/COVID19MEXICOsmall.csv")

# Set de train con el 70% de todos los datos
trainD = todosDatos.sample(frac=0.7)
# Set de test con el 30% restante de datos (excluimos los que ya están en train)
testD = todosDatos.drop(trainD.index)

# Separamos datos de entrada del modelo en el set de train
entradasTrain = torch.tensor(trainD.drop('DEF', axis = 1).values).type(torch.FloatTensor)
# Y el valor que deberá de predecir nuestro modelo
objetivoTrain = torch.tensor(trainD['DEF'].values).type(torch.FloatTensor)

# Hacemos lo mismo para el set de test
entradasTest = torch.tensor(testD.drop('DEF', axis = 1).values).type(torch.FloatTensor)
objetivoTest = torch.tensor(testD['DEF'].values).type(torch.FloatTensor)
#######################################################
############# DEFINICIÓN DEL MODELO ###################
#######################################################
# Creamos la arquitectura de la red utilizando la estructura de PyTorch
class Red(nn.Module):
    def __init__(self):
        super(Red, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # TODO 
        # Tal vez pueda ser de ayuda: https://pytorch.org/docs/stable/nn.html#linear-layers
        # Y esta otra: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

    def forward(self, x):
        # TODO
        return self.sigmoid(x)

# Creamos el modelo a partir de la arquitectura antes definida
red = Red()
# Definimos función de pérdida. Puedes encontrar, más aquí: https://pytorch.org/docs/stable/nn.html#loss-functions
lossFunction = nn.MSELoss()
# Definimos el optimizador y le indicamos qué paraámetros debe de actualizar. Puedes encontrar más aquí: https://pytorch.org/docs/stable/optim.html#algorithms
optimizer = optim.SGD(red.parameters(), lr=0.45)
#######################################################
############# ENTRENAMIENTO DEL MODELO ################
#######################################################
# Loop de entrenamiento #
epochs = 1000 # número de épocas a entrenar
for epoch in range(epochs):
    # Establecemos gradientes en cero
    optimizer.zero_grad()
    # Calculamos las predicciones de la red para los datos de entrada
    pred = red(entradasTrain)
    pred = pred.view(pred.size()[0]) # Reacomodamos para que tenga el mismo tamaño
    # Evaluamos el error entre las predicciones de la red y los objetivos reales
    perdida = lossFunction(pred, objetivoTrain)
    # Actualizamos los parámetros de la red
    perdida.backward() # Backpropagate
    optimizer.step() # Update params
    #print(f"epoch: {epoch} loss: {perdida}") # Muestra la pérdida en cada época
    # Gráfica inicial y final
    if epoch == 0 or epoch == epochs - 1:
        # Creamos un diccionario con los datos
        d = {'Obj': objetivoTrain.numpy(), 'Pred': pred.detach().numpy()}
        # Lo convertimos en un DataFrame
        df = pd.DataFrame(d)
        # Hacemos el boxplot de la predicción agrupado por el valor objetivo
        bp = df.boxplot(by="Obj", column="Pred")
        # Agregamos nombres de ejes y título
        [ax_tmp.set_xlabel('Objetivo') for ax_tmp in np.asarray(bp).reshape(-1)]
        [ax_tmp.set_ylabel('Predicción') for ax_tmp in np.asarray(bp).reshape(-1)]
        [ax_tmp.set_title('') for ax_tmp in np.asarray(bp).reshape(-1)]
        fig = np.asarray(bp).reshape(-1)[0].get_figure()
        fig.suptitle('Rendimiento en época ' + str(epoch + 1))
        # Mostramos la gráfica
        plt.show()
#######################################################
############# EVALUACIÓN DEL MODELO ###################
#######################################################
# Evaluamos en el set de test sin actualizar los gradientes de la red
with torch.no_grad():
    testPred = pred = red(entradasTest) # Obtenemos predicción
    testPred = testPred.view(testPred.size()[0])
    d = {'Obj': objetivoTest.numpy(), 'Pred': testPred.detach().numpy()}
    df = pd.DataFrame(d)
    bp = df.boxplot(by="Obj", column="Pred")
    [ax_tmp.set_xlabel('Objetivo') for ax_tmp in np.asarray(bp).reshape(-1)]
    [ax_tmp.set_ylabel('Predicción') for ax_tmp in np.asarray(bp).reshape(-1)]
    [ax_tmp.set_title('') for ax_tmp in np.asarray(bp).reshape(-1)]
    fig = np.asarray(bp).reshape(-1)[0].get_figure()
    fig.suptitle('Rendimiento set de test')
    plt.show()