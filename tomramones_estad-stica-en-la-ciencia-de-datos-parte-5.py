#Importamos los contextos

#Manera 1

from matplotlib import pyplot



#Manera 2

import matplotlib.pyplot as plt



#Nosotros usaremos la manera 1 para crear nuestros graficos

    #pyplot.plot(...)

#Elementos como por ejemplo: ejes, etiquetas y leyendas pueden ser accedidas y configuradas utilizando este contexto como funciones separadas.



#Podemos mostrar el grafico creado utilizando el siguiente comando

pyplot.show()



#Tambien podemos guardar el grafico en algun formato de nuestro interes

pyplot.savefig('imagen.png')

# Ejemplo de un grafio de linea

from numpy import sin

from matplotlib import pyplot

# creamos un intervalo constante para el eje X

x = [x*0.1 for x in range(100)]

# Creamos las observaciones para el eje Y, en este caso vamos a utilizar la funcion del sen

y = sin(x)

# utilizamos la funcion para implementar el grafio de linea

pyplot.plot(x, y)

# mostramos por pantalla el resultado

pyplot.show()
# Ejemplo de grafico de barras

from random import seed

from random import randint

from matplotlib import pyplot

# generamos una semilla aleatoria para poder reproducir nuestro codigo

seed(1)

# nombres de las categorias

x = ['red', 'green', 'blue']

# cantidades para cada categoria

y = [randint(0, 100), randint(0, 100), randint(0, 100)]

# creamos el grafico de barras

pyplot.bar(x, y)

# mostramos por pantalla el grafico

pyplot.show()
# Ejemplo de histograma

from numpy.random import seed

from numpy.random import randn

from matplotlib import pyplot

# generamos una semilla aleatoria para poder reproducir nuestro codigo

seed(13)

# generamos numeros aleatorias para una distribucion gaussiana.

x = randn(1000)

# creamos el grafico de histograma

pyplot.hist(x)

# mostramos por pantalla el grafico

pyplot.show()
# Ejemplo de box and whisker plot

from numpy.random import seed

from numpy.random import randn

from matplotlib import pyplot

# generamos una semilla aleatoria para poder reproducir nuestro codigo

seed(13)

# generamos numeros aleatorios respetando una distribucion normal

x = [randn(1000), 5 * randn(1000), 10 * randn(1000)]

# # creamos el grafico de box and whisker plot

pyplot.boxplot(x)

# mostramos por pantalla el grafico

pyplot.show()
# ejemplo de un scatter plot

from numpy.random import seed

from numpy.random import randn

from matplotlib import pyplot

# generamos una semilla aleatoria para poder reproducir nuestro codigo

seed(1)

# defino la variable x

x = 20 * randn(1000) + 100

# defino la variable y

y = x + (10 * randn(1000) + 50)

# creamos el grafico de scatter plot

pyplot.scatter(x, y)

# # mostramos por pantalla el grafico

pyplot.show()