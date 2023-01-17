# Cargamos las librerias que vamos a utilizar

from numpy.random import seed

from numpy.random import randn

from matplotlib import pyplot

# Generamos nuestra semilla de la suerte (esta funcion es para que los experimentos se puedan replicar con los mismos resultados, independientemente de donde se lo ejecute)

seed(1)

# Generamos observaciones univariadas

    #OBS: la funcion randn(n) genera una n cantidad de numeros al azar, siguiendo una distribucion gaussiana. Esto significa que el promedio sera CERO y

    # el desvio estandar sera de UNO. Ahora bien, podemos utilizar el metodo de reescalar para ajustar los valores a una distribucion gaussiana que sea de nuestro agrado

    # para ello podemos agregar un desvio estandar (5) y un promedio (50)

data = 5 * randn(10000) + 50

# Creamos un histograma

pyplot.hist(data,bins=100)

pyplot.show()
# vamos a calcular el promedio de las observaciones de la muestra

from numpy.random import seed

from numpy.random import randn

from numpy import mean



seed(1)



data = 5 * randn(10000) + 50



result = mean(data)

print('Mean: %.3f' % result)
# vamos a calcular la mediana de las observaciones de la muestra

from numpy.random import seed

from numpy.random import randn

from numpy import median



seed(1)



data = 5 * randn(10000) + 50



result = median(data)

print('Median: %.3f' % result)
# vamos a generar un grafico de distribucion gaussiana con diferentes tipos de varianza

from numpy import arange

from matplotlib import pyplot

from scipy.stats import norm

# eje x para el grafico

x_axis = arange(-3, 3, 0.001)

# grafico con varianza baja (color azul)

pyplot.plot(x_axis, norm.pdf(x_axis, 0, 0.5))

# grafico con varianza alta (color naranja)

pyplot.plot(x_axis, norm.pdf(x_axis, 0, 1))

pyplot.show()
# vamos a calcular la varianza de una muestra

from numpy.random import seed

from numpy.random import randn

from numpy import var



seed(1)

#generamos las observaciones univariadas

data = 5 * randn(10000) + 50

# calculamos la varianza

result = var(data)

print('Variance: %.3f' % result)
# vamos a calcular el desvio estandar de una muestra

from numpy.random import seed

from numpy.random import randn

from numpy import std



seed(1)

#generamos las observaciones univariadas

data = 5 * randn(10000) + 50

# calculamos el desvio estandar

result = std(data)

print('Standard Deviation: %.3f' % result)