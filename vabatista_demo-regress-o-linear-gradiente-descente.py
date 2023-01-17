#Importanto as bibliotecas e definindo padrões de saída
import math,sys,os,numpy as np
np.set_printoptions(precision=4, linewidth=100)
from numpy.random import random
from matplotlib import pyplot as plt, rcParams, animation, rc
from ipywidgets import interact, interactive, fixed
rcParams['figure.figsize'] = 8, 6
from time import sleep

# Função para a equação y = ax + b
def lin(a,b,x): 
    return a*x+b
a=3.
b=8.
np.random.seed(5)
n=30
x = random(n)
y = lin(a,b,x)
plt.scatter(x,y)
def sse(y,y_pred): 
    return ((y-y_pred)**2).sum()

def mse(y,a,b,x): 
    return sse(y, lin(a,b,x))/n
a_guess=-1.
b_guess=1.
mse(y, a_guess, b_guess, x)
# taxa de aprendizagem
lr=0.5
def back_propagation():
    global a_guess, b_guess
    y_pred = lin(a_guess, b_guess, x)
    dydb = 2 * (y_pred - y)
    dyda = x*dydb
    a_guess -= lr*dyda.mean()
    b_guess -= lr*dydb.mean()
%matplotlib notebook

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)
plt.ion()

fig.show()
fig.canvas.draw()

prev_error = 1000
a_guess=-5.
b_guess=10.

while prev_error - mse(y, a_guess, b_guess, x) > 0.001:
    prev_error = mse(y, a_guess, b_guess, x)
    back_propagation()
    
    ax.clear()
    ax.scatter(x,y)
    ax.plot(x,lin(a_guess,b_guess,x), 'r', label = 'a=%.4f, b=%.4f, mse=%.4f' % (a_guess, b_guess, prev_error))
    ax.legend(loc='upper left',prop={'size': 14})
    fig.canvas.draw()
    sleep(0.5)
