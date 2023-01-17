#importamos las librerias que vamos a utilizar

from numpy import arange

from matplotlib import pyplot

from scipy.stats import norm



# creamos el eje x del grafico

x_axis = arange(-5, 5, 0.01)



# creamos el eje y del grafico

y_axis = norm.pdf(x_axis, 0, 1)



# ploteamos el grafico

pyplot.plot(x_axis, y_axis)



# mostramos el grafico

pyplot.show()