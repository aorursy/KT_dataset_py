#CÉLULA PSO-LIB-01
import numpy as np
import math
import matplotlib.pyplot as plt
from enum import Enum
from operator import xor
%matplotlib inline
#CÉLULA PSO-LIB-03
def plot_population(Particles, generation):
    for p in Particles:
        plt.plot(p['X'], p['Y'], 'bo')
    plt.grid(True)
#Teste
particle1 = {'X': 84.43, 'Y': 93.11, 'XBest': 53.42, 'YBest': 68.11, 'VX': 54.83, 'VY': -67.83}
particle2 = {'X': 0, 'Y': 0, 'XBest': 53.42, 'YBest': 68.11, 'VX': 54.83, 'VY': -67.83}
particle3 = {'X': -80, 'Y': 80, 'XBest': -12, 'YBest': 1, 'VX': 54.83, 'VY': -67.83}

particules = (particle1, particle2, particle3)
plot_population(particules, 1)