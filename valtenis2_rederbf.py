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





import math

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
data = pd.read_excel('/kaggle/input/rbfrna/Tabel_Treinamento_RNA.xls')
X = np.array(data.drop('d', 1))

y = np.array(data['d'])
def grafDist(omg1,omg2,w1,w2, distXW1, distXW2):

    circle1 = plt.Circle(w1, np.max(distXW1), color='r', fill=False)

    circle2 = plt.Circle(w2, np.max(distXW2), color='blue', fill=False)

    fig, ax = plt.subplots()

    ax.plot(omg1[:,0],omg1[:,1], "yo",omg2[:,0],omg2[:,1], "co",w1[0],w1[1],"ro",w2[0],w2[1],"bo")

    ax.add_artist(circle1)

    ax.add_artist(circle2)
W1 = np.copy(X[0])

W2 = np.copy(X[1])
O1Ant = np.array([])

O2Ant = np.array([])



while(True):



    omega1 = []

    omega2 = []

    dew1s = np.array([])

    dew2s = np.array([])



    for i in range(len(X)):

        

        dew1 = math.sqrt((X[i][0]-W1[0])**2 + (X[i][1]-W1[1])**2)

        dew2 = math.sqrt((X[i][0]-W2[0])**2 + (X[i][1]-W2[1])**2)

        

        if(dew1 < dew2):

            omega1.append(X[i])

            dew1s = np.append(dew1s, dew1)

        else:

            omega2.append(X[i])

            dew2s = np.append(dew2s, dew2)

            



    omega1 = np.array(omega1)

    omega2 = np.array(omega2)



    if(np.array_equal(O1Ant, omega1) and np.array_equal(O2Ant, omega2)):

        grafDist(omega1, omega2, W1, W2, dew1s, dew2s)

        break

    else:

        O1Ant = np.copy(omega1)

        O2Ant = np.copy(omega2)

        

        W1[0] = omega1[:, 0].mean()

        W1[1] = omega1[:, 1].mean()



        W2[0] = omega2[:, 0].mean()

        W2[1] = omega2[:, 1].mean()
def somatoria(x, w):

    return sum((x - w)**2)
def variancia(x, w):

    return np.mean((x-w)**2)
var1 = variancia(O1Ant, W1)

var2 = variancia(O2Ant, W2)

print('var1 = {}\nvar2 = {}'.format(var1, var2))
W1
W2
def calcG(x, w, var):

    

    g = []

    

    for i in range(len(x)):

        g.append(math.exp(-somatoria(x[i], w)/(2*var)))

    

    return g
g1 = calcG(X, W1, var1)

g2 = calcG(X, W2, var2)
#Função para achar o EQM

def eqmFun(X,y,w):       

    yp=[]

    for i in range(len(X)):

        yp.append(np.dot(X[i],w))                        

    return mean_squared_error(y,np.array(yp))
X2 = np.array([g1, g2]).T

X2 = np.hstack((np.ones([len(X2), 1]), X2))
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.30, random_state=42)

_W = np.random.uniform(0, 1, len(X2[0]))

tolerancia=0.0000001

l_rate=0.01

num_epocas=100000

epocas=0

eqm=0.1

eqmIni=0.1
while(eqm > tolerancia):

           

    for i in range(len(X_train)): 

        yp= np.dot(X_train[i], _W)             #Previsao para calcular o erro         

        erro = y_train[i]-yp                   #Calculo do Erro

        _W += l_rate * erro * X_train[i]     #Atualiza os pesos

        

    eqmFim = eqmFun(X_train, y_train , _W)         #EQM "final" que sera utilisado para o calculo do EQM de parada do loop                           

    eqm = abs(eqmFim-eqmIni)           #Calculo do EQM que para o loop

    #print("EQM: {} | epocas: {}".format(eqm,epocas))

    

    eqmIni = eqmFim               #Copias o EQM "final" para o "inicial"

    epocas+=1                  #Adicona 1 ao numero de epocas

    if(epocas==num_epocas):     #Parada para quando atingir o maximo de epocas

        break
def predict(x, w):

    return sum(x * w), 1 if(sum(x * w)>=0) else -1
_W
def test_precisao(x, y, w):

    acertos = 0

    for i in range(len(x)):

        if(predict(x[i], w)[1] == y[i]):

            acertos += 1

    return "Total de Testes: {}, Acertos: {}, Erros: {}, Precisão: {} %".format(len(x), acertos, (len(x)-acertos), round((acertos*100)/len(x), 2))
test_precisao(X_train, y_train, _W)
test_precisao(X_test, y_test, _W)
data_teste = pd.read_csv('/kaggle/input/rbfrna/teste.csv')
xt = np.array(data_teste.drop('d', 1))

yt = np.array(data_teste['d'])
g1t = calcG(xt, W1, var1)

g2t = calcG(xt, W2, var2)
xt = np.hstack((np.ones([len(xt), 1]), np.array([g1t, g2t]).T))
test_precisao(xt, yt, _W)
for i in range(len(xt)):

    pred = predict(xt[i], _W)

    y = pred[0]

    ypos = pred[1]

    resp = yt[i]

    print('y = {:.7f} \t ypos = {} \t resp = {} \t {}'.format(y, ypos, resp, ypos==resp))