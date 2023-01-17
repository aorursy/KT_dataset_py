import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
def f(x):
    return x**2 - 2*x + 1

x = np.linspace(-5.0, 5.0, 100)
y = f(x)
plt.plot(x,y)
def f_prime(x):
    return 2*x - 2

x = np.linspace(-5.0, 5.0, 100)
y = f(x)
y_prime = f_prime(x)
plt.plot(x,y)
plt.plot(x,y_prime)
x = -4
print(f(x))
print(f_prime(x))
def f(x,y):
    return x**2 * y**2 + 2*x*y + y

from mpl_toolkits.mplot3d import Axes3D

def plot_f():
    x = np.linspace(-5.0, 5.0, 100)
    y = np.linspace(-5.0, 5.0, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X,Y)

    figure = plt.figure(1, figsize = (20, 10))
    subplot3d = plt.subplot(111, projection='3d')
    surface = subplot3d.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0.1)
    return subplot3d
plot_f()
plt.show()
def plot_point(subplot):
    plt.plot([4],[-4], [f(4, -4)], 'go')
    plt.plot([4],[-4], [0], 'gx')
    subplot.plot([4,4],[-4,-4],[f(4,-4), 0], color='gray', linestyle='dotted')

subplot = plot_f()
plot_point(subplot)    
from matplotlib.patches import Arrow
import mpl_toolkits.mplot3d.art3d as art3d

subplot = plot_f()
plot_point(subplot)

a = Arrow(4, -4, 0.71, -0.704, width=0.1, color='red')
subplot.add_patch(a)
art3d.pathpatch_2d_to_3d(a, z=0, zdir="z")

plt.show()
