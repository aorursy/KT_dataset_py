import matplotlib.pyplot as plt

import numpy as np

import random
nb_patterns = 4

pattern_width = 5

pattern_height = 9

n_pixels = pattern_width * pattern_height

max_iterations = 10
X = np.array([

    #1

    [

        -1, -1,  1,  1, -1,

        -1,  1,  1,  1, -1,

        -1, -1,  1,  1, -1,

        -1, -1,  1,  1, -1,

        -1, -1,  1,  1, -1,

        -1, -1,  1,  1, -1,

        -1, -1,  1,  1, -1,

        -1, -1,  1,  1, -1,

        -1, -1,  1,  1, -1

    ],

    #2

    [

         1,  1,  1,  1,  1,

         1,  1,  1,  1,  1,

        -1, -1, -1,  1,  1,

        -1, -1, -1,  1,  1,

         1,  1,  1,  1,  1,

         1,  1, -1, -1, -1,

         1,  1, -1, -1, -1,

         1,  1,  1,  1,  1,

         1,  1,  1,  1,  1

    ],

    #3

    [

         1,  1,  1,  1,  1,

         1,  1,  1,  1,  1,

        -1, -1, -1,  1,  1,

        -1, -1, -1,  1,  1,

         1,  1,  1,  1,  1,

        -1, -1, -1,  1,  1,

        -1, -1, -1,  1,  1,

         1,  1,  1,  1,  1,

         1,  1,  1,  1,  1

    ],

    #4

    [

         1,  1, -1,  1,  1,

         1,  1, -1,  1,  1,

         1,  1, -1,  1,  1,

         1,  1,  1,  1,  1,

         1,  1,  1,  1,  1,

        -1, -1, -1,  1,  1,

        -1, -1, -1,  1,  1,

        -1, -1, -1,  1,  1,

        -1, -1, -1,  1,  1

    ]

])
fig, ax = plt.subplots(1, nb_patterns, figsize=(9, 5))



fig.suptitle('Padrões Inseridos Para Treino', fontsize=16)



for i in range(nb_patterns):

    ax[i].matshow(X[i].reshape((pattern_height, pattern_width)), cmap='binary')

    ax[i].set_xticks([])

    ax[i].set_yticks([])



fig.tight_layout()

plt.show()
W = np.zeros((n_pixels, n_pixels))

MId = np.identity(n_pixels)



for i in range(n_pixels):

    for j in range(n_pixels):

        

        #1p

        w = 0.0

        for n in range(nb_patterns):

            w += X[n][i] * X[n][j]

            

        W[i][j] += w / X.shape[0]

        

        #2p

        W[i][j] -= MId[i][j] * (nb_patterns/n_pixels)

W
def gerarRuido(num, porc):

    

    vet = np.copy(num)

    size = len(vet)

    qtdPixels = (size * porc) // 100

    

    pixelsToModify = random.sample(range(size), qtdPixels)



    vet[pixelsToModify] *= -1 

            

    return vet
def g(u):

    return 1.0 if (u > 0) else -1.0



def recoveryPattern(patternCorrupted):



    v_atual = patternCorrupted.copy()



    while(True):

        

        v_anterior = v_atual.copy()

        

        for i in range(pattern_width * pattern_height):

            u = np.dot(W[i], v_atual)

            v_atual[i] = g(u)

            

        if(np.array_equal(v_atual, v_anterior)):

            break

        

    return v_atual
def showPatterns(titlePattern, patternOriginal, patternCorrupted, patternRecovered):

    

    fig, ax = plt.subplots(1, 3, figsize=(9, 5))

    fig.suptitle(titlePattern, fontsize=16)



    subtitles = ['Imagem Transmitida \n (sem ruídos)', 'Imagem Recebida \n (com ruídos)', 'Imagem Recuperada \n (sem ruídos)']

    patterns = np.array([patternOriginal, patternCorrupted, patternRecovered])

    

    for i in range(len(patterns)):

        ax[i].matshow(patterns[i].reshape(pattern_height, pattern_width), cmap='binary')

        ax[i].set_title(subtitles[i])

        ax[i].set_xticks([])

        ax[i].set_yticks([])

    

    fig.tight_layout()

    fig.subplots_adjust(top=0.75)

    

    plt.show()
def testar(titlePattern, padrao, ruidoPorc):

    

    #Padrão escolhido

    patternOriginal = padrao.copy()

    

    #Cria o padrão com ruídos

    patternCorrupted = gerarRuido(patternOriginal, ruidoPorc)

    

    #Recupera o padrão original usando a rede hopefield

    patternRecovered = recoveryPattern(patternCorrupted)

    

    #Mostrando os resultados

    showPatterns(titlePattern, patternOriginal, patternCorrupted, patternRecovered)
for i in range(3):

    testar('Padrão Número 1 com 20% de ruído', X[0], 20)
for i in range(3):

    testar('Padrão Número 2 com 20% de ruído', X[1], 20)
for i in range(3):

    testar('Padrão Número 3 com 20% de ruído', X[2], 20)
for i in range(3):

    testar('Padrão Número 4 com 20% de ruído', X[3], 20)
testar('Padrão Número 1 com 40% de ruído', X[0], 40)
testar('Padrão Número 2 com 40% de ruído', X[1], 40)
testar('Padrão Número 3 com 40% de ruído', X[2], 40)
testar('Padrão Número 4 com 40% de ruído', X[3], 40)
testar('Padrão Número 1 com 50% de ruído', X[0], 50)
testar('Padrão Número 2 com 50% de ruído', X[1], 50)
testar('Padrão Número 3 com 50% de ruído', X[2], 50)
testar('Padrão Número 4 com 50% de ruído', X[3], 50)