#Importanto as bibliotecas e definindo padrões de saída

%matplotlib inline
import math,sys,os,numpy as np
from numpy.random import random
from matplotlib import pyplot as plt, rcParams, animation, rc
from __future__ import print_function, division
from ipywidgets import interact, interactive, fixed
from ipywidgets.widgets import *
rc('animation', html='html5')
rcParams['figure.figsize'] = 3, 3
%precision 4
np.set_printoptions(precision=4, linewidth=100)
# Função para a equação y = ax + b
def lin(a,b,x): 
    return a*x+b
a=3.
b=8.
n=30
x = random(n)
y = lin(a,b,x)
x
y
plt.scatter(x,y)
def sse(y,y_pred): return ((y-y_pred)**2).sum()
def loss(y,a,b,x): return sse(y, lin(a,b,x))
def mre(y,a,b,x): return np.sqrt(loss(y,a,b,x)/n)
a_guess=-1.
b_guess=1.
mre(y, a_guess, b_guess, x)
lr=1
def back_propagation():
    global a_guess, b_guess
    y_pred = lin(a_guess, b_guess, x)
    dydb = 2 * (y_pred - y)
    dyda = x*dydb
    a_guess -= lr*dyda.mean()
    b_guess -= lr*dydb.mean()
back_propagation()
plt.scatter(x,y)
line, = plt.plot(x,lin(a_guess,b_guess,x))
print(a_guess, b_guess, mre(y,x,a_guess, b_guess))

