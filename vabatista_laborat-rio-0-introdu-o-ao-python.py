def quicksort(arr):

    if len(arr) <= 1:

        return arr

    pivot = arr[int(len(arr) / 2)]

    left = [x for x in arr if x < pivot]

    middle = [x for x in arr if x == pivot]

    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)



print(quicksort([3,6,8,10,1,2,1]))
x = 3

print(x, type(x))
print(x + 1   )  # adição;

print(x - 1   )  # subtração

print(x * 2   )  # multiplicação

print(x ** 2  )  # potência
print(7 // 3)  # divisão inteira de 7 por 3

print(7 % 3)   # resto da divisão inteira de 7 por 3
y = 2.5 ## para definir números reais, usar o '.'

print(type(y) )  # imprime o tipo

print(y, y + 1, y * 2, y ** 2 )  # imprime "2.5 3.5 5.0 6.25"
t, f = True, False # designação dupla de variáveis. t recebe True e f recebe False

print(type(t) )  # Imprime "<type 'bool'>"
print(t and f )  # Logical AND;

print(t or f  )  # Logical OR;

print(not t   )  # Logical NOT;

print(t != f  )  # Logical XOR;
hello = 'hello'   # Strings usam aspas simples

world = "world"   # ou duplas, não importa

print(hello, len(hello) )  # a vírgula pode ser usada para encadear (concatenar) várias strings ao imprimir
hw = hello + ' ' + world  # Ou use a operação de adição para concatenar

print(hw  )  # imprime "hello world"
hw12 = '%s %s %d %.4f' % (hello, world, 12, 10)  # impressão no estilo printf do C++

print(hw12  )  # imprime  "hello world 12"
s = "hello"

print(s.capitalize()  )

print(s.upper()       )

print(s.rjust(7)      )  # Alinha a direita num texto de tamanho 7

print(s.center(7)     )  # Centraliza

print(s.replace('l', '(ell)')  )  # substitui todas as instâncias de uma substring por outra

print('  world '.strip()  )  # remove espaçõs antes e depois
xs = [3, 1, 2]   # cria uma lista

print(xs, xs[2])

print(xs[-1], xs[-2], xs[-3] )  # indices negativos contam do fim para o início da lista, mas começam em 1 em vez de 0
xs[2] = 'foo'    # Atribui um novo valor

print(xs)
xs.append('bar') # Acrescenta um novo item

print(xs  )
x = xs.pop()     # Remove e retorna o último elemento

print(x, xs )
nums = list(range(5))    # range é uma função que cria uma lista

print(nums         )  # imprime "[0, 1, 2, 3, 4]"

print(nums[2:4]    )  # slice do indice 2 ao 4 (exclusivo); imprime "[2, 3]"

print(nums[2:]     )  # slice do índice 2 até o fim da lista; imprime "[2, 3, 4]"

print(nums[:2]     )  # slice do início da lista até o índice 2 (exclusivo); imprime "[0, 1]"

print(nums[:]      )  # : sem os índices significa todos os índices. Para listas com mais de uma dimensão tem grande utilidade

print(nums[:-1]    )  # índices do slice pode ser negativo; imprime ["0, 1, 2, 3]"

nums[2:4] = [8, 9] # atribui uma nova sublista ao slice

print(nums         )  # imprime "[0, 1, 8, 9, 4]"
animals = ['cat', 'dog', 'monkey']

for animal in animals:

    print(animal)
animals = ['cat', 'dog', 'monkey']

for idx, animal in enumerate(animals):

    print('#%d: %s' % (idx + 1, animal))
nums = [0, 1, 2, 3, 4]

squares = []

for x in nums:

    squares.append(x ** 2)

print(squares)
nums = [0, 1, 2, 3, 4]

squares = [x ** 2 for x in nums]

print(squares)
nums = [0, 1, 2, 3, 4]

even_squares = [x ** 2 for x in nums if x % 2 == 0]

print(even_squares)
d = {'cat': 'cute', 'dog': 'furry'}  # Cria o dicionário. ',' separa a relação de pares de chave ':' valor

print(d['cat']       )  # recupera uma entrada pela chave

print('cat' in d     )  # Verifica se uma chave está presente no dicionário.
d['fish'] = 'wet'    # Atribui uma entrada nova no dicionário. Sobrescreve caso já exista

print(d['fish']      )  # Imprime "wet"
print(d['monkey']  )  # KeyError: 'monkey' não está no dicionário
print(d.get('monkey', 'N/A')  )  # Recupera um elemento com um valor default caso não exita, evitando KeyError; prints "N/A"

print(d.get('fish', 'N/A')    )  # Aqui o elemento foi encontrado
del d['fish']        # Remove um elemento do dicionário

print(d.get('fish', 'N/A') )  # "fish" não está mais no dicionário
d = {'person': 2, 'cat': 4, 'spider': 8}

for animal in d:

    legs = d[animal]

    print('A %s has %d legs' % (animal, legs))
d = {'person': 2, 'cat': 4, 'spider': 8}

for (animal, legs) in d.items():

    print('A %s has %d legs' % (animal, legs))
nums = [0, 1, 2, 3, 4]

even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}

print(even_num_to_square)
animals = {'cat', 'dog'}

print('cat' in animals   )  # Verifica a existência de um elemento

print('fish' in animals  )

animals.add('fish') #adiciona novo elemento

print(len(animals) )  #imprime a quantidade de elementos

animals.add('cat')       # adicionar um elemento existente não muda o conjunto

print(len(animals)       )
animals = {'cat', 'dog', 'fish'}

for idx, animal in enumerate(animals):

    print('#%d: %s' % (idx + 1, animal))

# Prints "#1: fish", "#2: dog", "#3: cat"
from math import sqrt

print({int(sqrt(x)) for x in range(30)})
d = {(x, x + 1): x for x in range(10)}  # Cria um dicionário cujas chaves são tuplas

t = (5, 6)       # Cria uma tupla

print(type(t))

print(d)

print(d[t]       )

print(d[(1, 2)] )  #recupera o valor do elemento cuja chave é (1,2)
t[0] = 1 # tuplas não suportam atribuição de itens, pois são imutáveis.
def sign(x):

    if x > 0:

        return 'positive'

    elif x < 0:

        return 'negative'

    else:

        return 'zero'



for x in [-1, 0, 1]:

    print(sign(x))
def hello(name, loud=False):

    if loud:

        print('HELLO, %s' % name.upper())

    else:

        print('Hello, %s!' % name)



hello('Bob')

hello('Fred', loud=True)
import numpy as np
a = np.array([1, 2, 3])  # cria um array de rank 1 

print(type(a), a.shape, a[0], a[1], a[2])

a[0] = 5                 # Muda o elmento de um array

print(a                  )
b = np.array([[1,2,3],[4,5,6]])   # Cria um array de rank 2

print(b)

print(b.shape                   )

print(b[0, 0], b[0, 1], b[1, 0])
a = np.zeros((2,2))  # Array de zeros

print(a)
b = np.ones((1,2))   # Array de "um's"

print(b)
c = np.full((2,2), 7) # Array de uma constante qualquer

print(c )



# O alerta mostra que deveríamos explicitamente declarar o número de ponto flutuante, pois em versões futuras será considerado um

#array de inteiros.
d = np.eye(2)  # Cria um array (matriz) identidade 2x2 

print(d)
e = np.random.random((2,2)) # Cria uma matrix 2x2 com valores aleatórios

print(e)
import numpy as np



# Cria um array de rank 2 de dimensões (3, 4)

# [[ 1  2  3  4]

#  [ 5  6  7  8]

#  [ 9 10 11 12]]

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])



# Cria um novo array tirando uma fatia das duas primeiras linhas e colunas 1 e 2 (2a e 3a colunas):

# [[2 3]

#  [6 7]]

b = a[:2, 1:3]

print(b)
print(a[0, 1]  )

b[0, 0] = 77    # b[0, 0] é o mesmo dado (mesma posição no vetor original) que a[0, 1]

print(a[0, 1] )
a = np.array([[1,2], [3, 4], [5, 6]])

print(a)

# O resultado é um array de formato (3,)

print(a[[0, 1, 2], [0, 1, 0]])



# Ou poderíamos fazer o mesmo da seguinte forma:

print(np.array([a[0, 0], a[1, 1], a[2, 0]]))
import numpy as np



a = np.array([[1,2], [3, 4], [5, 6]])

print(a)

bool_idx = (a > 2)  # Retorna um array de booleanos 



print(bool_idx)
# podemos usar a técnica para construir um array de rank 1 que consiste dos elementos que atendem ao critério

print(a[bool_idx])
x = np.array([1, 2])  # Numpy escolhe o tipo

y = np.array([1.0, 2.0])  # Numpy escolhe o tipo

w = np.array([1, 2], dtype=np.int32)  # Força o tipo

z = np.array([1.0, 2.0], dtype=np.int32)  # Força o tipo. Nesse caso há conversão de tipos.



print(x.dtype, y.dtype, w.dtype, z.dtype)
%%time

a = range(100000000) ## lista python (built-in)

print(type(a))

print(sum(a))
%%time

a = np.arange(100000000) #array numpy

print(type(a))

print(np.sum(a))
x = np.array([[1,2],[3,4]], dtype=np.float64)

y = np.array([[5,6],[7,8]], dtype=np.float64)



# Soma elemento por elmento em cada posição

print(x + y)

print(np.add(x, y))
print(x - y)

print(np.subtract(x, y))
print(x * y)

print(np.multiply(x, y))
print(x / y)

print(np.divide(x, y))
print(np.sqrt(x))
x = np.array([[1,2],[3,4]])

y = np.array([[5,6],[7,8]])



v = np.array([9,10])

w = np.array([11, 12])



# Produto interno dos vetores

print(v.dot(w))

print(np.dot(v, w))
# Multiplicação da matriz x pelo vetor v

print(x.dot(v) )

print(np.dot(x, v))
# Multiplicação da matriz x pela y

print(x.dot(y))

print(np.dot(x, y))
x = np.array([[1,2],[3,4]])

print(x)

print(np.sum(x)  )  # Soma todos os elmentos do array em todas as dimensões

print(np.sum(x, axis=0)  )  # Soma as colunas

print(np.sum(x, axis=1)  )  # Soma as linhas
print(x)

print(x.T )  # transposta

print(x.reshape(1,4))

a = np.array([[1,2,3]])

print(a.T)
import matplotlib.pyplot as plt # importa o módulo e o apelida de plt
%matplotlib inline
x = np.arange(0, 3 * np.pi, 0.1) # gera números entre 0 e 3*PI, incrementando de 0.1 em 0.1

y_sin = np.sin(x) # calcula o seno de cada elemento de X



# Gera o gráfico de sen(x)

plt.plot(x, y_sin)
y_cos = np.cos(x)



plt.plot(x, y_sin)

plt.plot(x, y_cos)

plt.xlabel('x axis label')

plt.ylabel('y axis label')

plt.title('Sine and Cosine')

plt.legend(['Sine', 'Cosine'])
# Calcula seno e cosseno de X

x = np.arange(0, 3 * np.pi, 0.1)

y_sin = np.sin(x)

y_cos = np.cos(x)



# Configura um grid de gráficos que tem 2 linhas e uma coluna

# e diz que vamos trabalhar no primeiro elemento (primeiro gráfico)

plt.subplot(2, 1, 1)



# Gera o primerio gráfico

plt.plot(x, y_sin)

plt.title('Sine')



# Muda para o segundo subplot e gera o segundo gráfico

plt.subplot(2, 1, 2)

plt.plot(x, y_cos)

plt.title('Cosine')



# Mostra a figura

plt.show()
# Importando a biblioteca e apelidando de pd

import pandas as pd
# Cria um índice de datas

dates = pd.date_range('20130101', periods=6)



# Cria uma série

serie = pd.Series([1,3,5,np.nan,6,8], index=dates)

print(serie)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

print(df)
df2 = pd.DataFrame({ 'A' : 1.,

   'B' : pd.Timestamp('20130102'),

   'C' : pd.Series(1,index=list(range(4)),dtype='float32'),

   'D' : np.array([3] * 4,dtype='int32'),

   'E' : pd.Categorical(["test","train","test","train"]),

  'F' : 'foo' })

print(df2)
#dtypes retorna o tipo de cada coluna

df2.dtypes
df.head() #notar que sem usar print, a formatação da saída fica como tabela.
df.values
df.describe()
# Ordenando pelo índice

df.sort_index(ascending=False).head()
# Ordenando pelos valores

df.sort_values(by='B')
# Exibindo uma única coluna.

df['A'] # ou o equivalente df.A
# Selecionando por posição

df[0:3]
# selecionando pelo índice

df['20130102':'20130104'] # note que é uma seleção inclusiva, diferente da seleção pela posição
dfParasite = pd.read_csv('../input/lab0_parasite_data.csv')
dfParasite.head()
dfParasite.describe()