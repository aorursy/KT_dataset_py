# Importing libraries

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import math



from mpl_toolkits.mplot3d import Axes3D



from pylab import meshgrid, imshow, contour, colorbar
# Loading data

df = pd.read_csv('../input/santa-workshop-tour-2019/family_data.csv')
# Function that calculates exponntial part of function

def expon(x, y, log = False):

    if log:        

        return np.log1p((x**(0.5+abs(x-y)/50)))

    else:

        return (x**(0.5+abs(x-y)/50))



# Defining x and y

x = range(125, 301)

y = range(125, 301)

X, Y = meshgrid(x, y)



# Plot behaviour of each part of function

fig = plt.figure(figsize = (20, 5))

rows, cols = (1, 3)



fig.add_subplot(rows, cols, 1)

x = range(125, 301)

y = [(i-125) / 400 for i in x]

plt.plot(x, y)

plt.grid(); plt.title('Linear part'); plt.xlabel('Days'); plt.ylabel('Factor')



Z = expon(X, Y)

ax = fig.add_subplot(rows, cols, 2, projection = '3d')

surf = ax.plot_surface(X, Y, Z, cmap = 'coolwarm')

ax.view_init(30, 10)

ax.set_xlabel('Nd')

ax.set_ylabel('Nd+1')

ax.set_zlabel('Penalty')

ax.set_title('Exponential part')



Z = expon(X, Y, log = True)

ax = fig.add_subplot(rows, cols, 3, projection = '3d')

surf = ax.plot_surface(X, Y, Z, cmap = 'coolwarm')

ax.view_init(30, 35)

ax.set_xlabel('Nd')

ax.set_ylabel('Nd+1')

ax.set_zlabel('Penalty')

ax.set_title('Log1p exponential part')



plt.show()
def penalty(x, y, log = False):

    if log:        

        return np.log1p(((x-125)/400) * (x**(0.5+abs(x-y)/50)))

    else:

        return ((x-125)/400) * (x**(0.5+abs(x-y)/50))



x = range(125, 301)

y = range(125, 301)

X, Y = meshgrid(x, y)





rows = 2

cols = 2



fig = plt.figure(figsize = (20, 20))



Z = penalty(X, Y)

ax = fig.add_subplot(rows, cols, 1, projection = '3d')

surf = ax.plot_surface(X, Y, Z, cmap = 'coolwarm')

ax.view_init(30, 35)

ax.set_xlabel('Nd')

ax.set_ylabel('Nd+1')

ax.set_zlabel('Penalty')

ax.set_title('Penalty function 3D plot')



fig.add_subplot(rows, cols, 2)

plt.contourf(X, Y, Z, cmap = 'coolwarm')

plt.xlabel('Nd')

plt.ylabel('Nd+1')

plt.title('Penalty function 2D plot')



Z_log = penalty(X, Y, log = True)

ax = fig.add_subplot(rows, cols, 3, projection = '3d')

surf = ax.plot_surface(X, Y, Z_log, cmap = 'coolwarm')

ax.view_init(30, 35)

ax.set_xlabel('Nd')

ax.set_ylabel('Nd+1')

ax.set_zlabel('Penalty')

ax.set_title('log1p Penalty function 3D plot')



fig.add_subplot(rows, cols, 4)

plt.contourf(X, Y, Z_log, cmap = 'coolwarm')

plt.xlabel('Nd')

plt.ylabel('Nd+1')

plt.title('log1p Penalty function 2D plot')



plt.show()
fig = plt.figure(figsize = (20, 10))



plt.subplot(121)

for i in range(X.shape[0]):

    plt.plot(X[i], Z[i])

plt.xlabel('Nd')

plt.ylabel('Penalty')

plt.title('Penalty function for each Nd+1 value')

    

plt.subplot(122)

for i in range(X.shape[0]):

    plt.plot(X[i], Z_log[i])    

plt.xlabel('Nd')

plt.ylabel('Penalty')

plt.title('log1p Penalty function for each Nd+1 value')



plt.show()
print(f'Maximum value of penalty function: {max([max(i) for i in Z])}')

print(f'Minimum value of penalty function: {min([min(i) for i in Z])}')



equal = []

for i, j in enumerate(Z):

    equal.append(j[i])

    

plt.plot(x, equal)

plt.xlabel('Nd == Nd+1')

plt.ylabel('Penalty')

plt.title('Penalty if Nd = Nd+1')

plt.show()