from numpy import *

A = matrix( [[1,2], [3,4]] )  #cria uma matriz de duas linhas e duas colunas. A primeira linha terá 1 e 2, e a segunda linha terá 3 e 4

print(A)
#para acessar um elemento da matriz, usamos apenas um par de colchetes:

A[0,1]   #acessa o elemento na linha 0 e coluna 1
#podemos acessar uma linha:

A[0]
#é preciso ter cuidado. Ao fazer a atribuição de linha, não se gera cópia:

linha = A[1]   #linha passa a apontar para a linha 1 de A (não gera cópia)

linha[0, 1] = 7   #alterará a linha 1 de A



print(A)
#para copiar, pode-se usar o método copy (aqui o operador de fatia não funciona para fazer cópia)

linha = A[1].copy()     #agora linha aponta para uma cópia da linha 1 de A, e pode ser alterada sem alterar A

linha[0, 0] = -5

print("A: ", A)

print("linha: ", linha)
#acessando uma coluna

A[:, 1]   #acessa a coluna 1
z = zeros( (3,5) )    #gera um nd array com 3 linhas e 5 colunas de zeros

print(z)
#sempre que geramos um array bidemensional, podemos fazer a conversão para matrix

M = matrix( zeros( (4,5) ) )    #gera matriz com 4 linhas e 5 colunas de zeros

print(M)
P = matrix( ones( (4,6) ) )  #Gera uma matriz 4x6 de 1's

print(P)
#podemos gerar matrizes de números aleatórios usando o submódulo random:

from numpy.random import *

N = matrix( randint(-10, 10, (2,3)) )   #gera uma matriz com 2x3 de números aleatórios entre -10 e 10

print(N)
#podemos trabalhar com submatrizes

M[1:3, 1:4] = N     #Atribui os coeficientes de N a uma submatriz de M  (dá erro se as dimensões não baterem)

print(M)
#podemos usar o atributo shape para redimensionar uma matriz.

#Por exemplo, sendo M uma matriz 4x5, podemos modificá-la para 2x10

M.shape = (2,10)  #redimensiona a matriz M

print(M)
#A função eye gera um array identidade

B  = matrix( eye(2,2),  int )  #gera uma matriz indentidade 2x2. Aqui, especificamos que o tipo é int em vez de float

print(B)
#podemos multiplicar uma matriz por um um escalar:

B = 4*B

print(B)
#somar matrizes

C = A + B

print(C)
#multiplicá-las

D = A*B

print(D)
#fazer potenciação

E = A**2   #equivalente a fazer A*A

print(E)
#usando a potenciação, podemos calcular a inversa de uma matriz

Ainv = A**-1

print("Ainv: ", Ainv)
#multiplica uma matriz por sua inversa deveria resultar na matriz identidade:

I = A*Ainv

print(I)
transpose(A)  #gera matriz transposta de A
trace(A)   #obtém o traço da matriz A
#a função hstack concatena matrizes horizontalmente (por coluna)

hstack( (A,B) )
#a função vstack concatena matrizes verticalmente (por lina)

vstack( (A,B) )
sqrt(A)  #gera matriz com raizes quadradas dos coeficientes de A
#soma dos elementos

sum(A)
#pode-se conferir as diversas funções do módulo numpy olhando sua documentação

import numpy

help(numpy)
from numpy.linalg import *  #importando o submódulo de álgebra linear do numpy
#a função eig calcula autovalores e autovetores de uma matriz.

autovals, autovets = eig(A)    #obtém autovalores e autovetores de A. os autovetroes estão nas colunas de autovets



print('autovalores: ', autovals)

print('autovetores (colunas): ', autovets)
#A função det de numpy.linalg calcula o determinante de uma matriz

det(A)
matrix_rank(A)
b = matrix( [ [10], [20] ] )
#resolve o sistema Ax = b

x = solve(A, b)

print(x)
import numpy.linalg as la

help(la)
#plotando o gráfico resultante da ligação dos pontos A(-2, 40), B(0, 0) e C(5, 60)



import matplotlib.pyplot as plt

x = [-2, 0, 5]

y = [40, 0, 60]



plt.plot(x, y)

plt.show()
#plotando o gráfico de e^x no intervalo 1 <= x <= 5 com passo 0.2

import numpy as np

import matplotlib.pyplot as plt



x2 = np.arange(1.0, 5.1, 0.2)   #função numpy similar a função range, mas que funciona tb com floats

y2 = np.exp(x2)



plt.title('Gráfico exponencial')  #define um título



plt.plot(x2, y2, 'g')  #a opção g especifica que a curva plotada será verde

plt.show()
#vamos plotar uma superfície em 3 dimensões

#gráfico do paraboloide hiperbólico z = (-1/6)x^2  + (1/4)y**2

#para x e y no intervalo [-20  20]. Vamos usar passo 0.5



from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

import matplotlib.pyplot as plt

import numpy as np



fig = plt.figure()

ax = fig.gca(projection='3d')

x = np.arange(-20, 20, 0.5)

y = np.arange(-20, 20, 0.5)



X,Y = np.meshgrid(x, y)   #gera matrizes X e Y que representarão o produto cartesiano x por y.

Z = -(X**2)/6 + (Y**2)/4



surf = ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.jet, linewidth = 0)

fig.colorbar(surf)



plt.show()