import numpy as np

import pandas as pd

from math import sqrt

import matplotlib.pyplot as plt
n_entradas = 3

n_linhas = 4

n_colunas = 4

n_neuronios = 16

l_rate = 0.001

min_change = 0.0001

max_epochs = 2000
def obterVizinhos(shape, raio):

    

    linhas, colunas = shape

    

    mapNeuron = []

    num = 1

    for i in range(linhas):

        linha = []

        for j in range(colunas):

            linha.append(num)

            num += 1

        mapNeuron.append(linha)

    

    mapVizinhos = []

    

    for x1 in range(linhas):

        for y1 in range(colunas):

            vizin = []

            for x2 in range(linhas):

                for y2 in range(colunas):

                    if(mapNeuron[x2][y2] != mapNeuron[x1][y1]):

                        distEucli = sqrt((x1-x2)**2 + (y1-y2)**2)

                        if(distEucli <= raio):

                            vizin.append(mapNeuron[x2][y2])

            mapVizinhos.append(vizin)

            

    return mapVizinhos
def normalizar(vetor):

    vet = vetor.copy()

    norm = np.linalg.norm(vet)

    return vet/norm
def minDistance(X, W):

    dists = np.array([])

    

    for n in range(n_neuronios):

        dists = np.append(dists, np.linalg.norm(X-W[n]))

        

    return np.argmin(dists)
def getClassesNeurons(X, W):

    

    winners = np.array([])



    for i in range(len(X)):

        winners = np.append(winners, minDistance(X[i], W))



    nClasses = ['' for _ in range(n_neuronios)]



    for i in range(n_neuronios):

        classe = ''

        if(i in winners[:20]):

            classe = 'A'

        if(i in winners[20:60]):

            classe = 'B'

        if(i in winners[61:]):

            classe = 'C'

        nClasses[i] = classe



    return nClasses
def showGraph(X, W, title):

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')



    xcA = X[:20, 0]

    ycA = X[:20, 1]

    ycA = X[:20, 2]



    xcB = X[20:60, 0]

    ycB = X[20:60, 1]

    ycB = X[20:60, 2]



    xcC = X[61:, 0]

    ycC = X[61:, 1]

    ycC = X[61:, 2]



    ax.scatter(xcA, ycA, ycA, c='r', marker='s')

    ax.scatter(xcB, ycB, ycB, c='b', marker='^')

    ax.scatter(xcC, ycC, ycC, c='y', marker='x')

    

    classes = getClassesNeurons(X, W)

    

    for i in range(n_neuronios):

        

        color = 'k'

        sizePoint = 200 if classes[i] == '' else 1000

            

        if(classes[i] == 'A'):

            color = 'r'

        if(classes[i] == 'B'):

            color = 'b' 

        if(classes[i] == 'C'):

            color = 'y' 

        

        ax.scatter(W[i][0], W[i][1], W[i][2], s=sizePoint, c=color, marker='.')



    ax.set_xlabel('Eixo X')

    ax.set_ylabel('Eixo Y')

    ax.set_zlabel('Eixo Z')

    

    fig.set_size_inches(7, 7, forward=True)

    plt.title(title, fontsize=28)

    plt.gca().legend(('Classe A','Classe B', 'Classe C', 'Neurônios'))

    plt.show()
def initW(X):

    W = np.random.uniform(0,1,(n_neuronios, n_entradas))

    for i in range(n_neuronios):

        W[i] = X[i].copy()

    showGraph(X, W, 'Estado Inicial')

    return W
def train(X):

    

    X = normalizar(X)

    W = initW(X)

    

    vizinhos = obterVizinhos(shape=(n_linhas, n_colunas), raio=1)

    epochs = 0



    while(True):

        change = 0.0

        for i in range(len(X)):

            win = minDistance(X[i], W)

            

            W[win] += l_rate * (X[i]-W[win])

            change += l_rate * (X[i]-W[win])

            

            for viz in vizinhos[win]:

                W[viz-1] += (l_rate/2)*(X[i]-W[viz-1])

                W[viz-1] = normalizar(W[viz-1])

                

        change = sum(change)

        epochs += 1

        if(abs(change) <= min_change or epochs == max_epochs):

            break

    

    print('Épocas: ', epochs)

    

    return W, vizinhos
def predict(X, W, nClasses):

    return nClasses[minDistance(X, W)]
X = np.array(pd.read_csv('/kaggle/input/somsom/train.csv'))

X
W, nViz = train(X)

nClasses = getClassesNeurons(X, W)

showGraph(X, W, 'Estado Final')
for i in range(n_neuronios):

    print(' [Neurônio {:2d}]'.format(i+1))

    print(' Vizinhos: {}'.format(nViz[i]))

    print(' Pesos: {}'.format(W[i]))

    print(' Classe: {}'.format("''" if nClasses[i] == '' else nClasses[i]))

    print('-'*50)
for i in range(len(X)):

    print('X[{}] = {}'.format(i, predict(X[i], W, nClasses)))
Xtest = np.array(pd.read_csv('/kaggle/input/somsom/test.csv'))

Xtest
for i in range(len(Xtest)):

    print('Xtest[{}] = {}'.format(i, predict(Xtest[i], W, nClasses)))