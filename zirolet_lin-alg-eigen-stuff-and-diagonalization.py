# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Here we will just create some random vector

v = np.array([1,2])

#Let's display this vector in matplotlib

plt.quiver(v[0],v[1], angles='xy',scale_units='xy', scale = 1)

plt.xlim(-5,5)

plt.ylim(-5,5)

plt.grid()

plt.show()
#Now let's make a simple 2-D matrix to mess with

A =  np.array([[1,0],[0,2]])

A
from sympy import * #Used to place "Placeholders" and print steps to solve using characteristic polynomial

def findeigenvalues(A): #where A is a matrix

    #Let's get A-lambda(I) matrix ready:

        n = A.shape[1]

        I = Matrix(np.identity(2,dtype=int)) #I

        a = Symbol('a') #lambda

        S = Matrix(A) #A

        for i in range(n):

            print((S-a*I)[i,:])

        cpoly = (S - a*I).det() #det(A-lambda(I))

        print("p(a) = " , cpoly)

        eigenvalues = solve(cpoly)

        print("The eigenvalues are:" , eigenvalues)

        return eigenvalues

print(findeigenvalues(A))
#Or we can use numpy's own eigenvalue finder that uses an eigenvalue algorithm:

np.linalg.eig(A)[0] #We choose index 0, because that contains an np.array with the eigenvalues
def findeigenvectors(A):

    eigenvalues = findeigenvalues(A)

    print()

    zeros = np.zeros((1,A.shape[0]),dtype=int)

    for i in range(len(eigenvalues)):

        matrix = A-eigenvalues[i]*np.identity(2,dtype=int)

        print("Matrix -", eigenvalues[i],":")

        print(matrix) 

        #Note, using np.linalg.solve will not work, because we can see that when lambda = 1, there is not full rank. So this works only when there is full rank

        #np.linalg.solve(matrix,zeros)

        print()

findeigenvectors(A)
#Or the best method, which is to use np.linalg.eig(A)[1] to grab all of the eigenvectors

eigenvectors = np.linalg.eig(A)[1]

#eigenvectors[:,i] will have the eigenvector for eigenvalue[i].

#So in our case,

val1 = np.linalg.eig(A)[0][0]

vec1 = eigenvectors[:,0]

val2 = np.linalg.eig(A)[0][1]

vec2 = eigenvectors[:,1]

print("Eigenvector for lambda =", val1, ":" , vec1)

print("Eigenvector for lambda =", val2, ":" , vec2)
#Our non-eigenvectors that will "shift"

x0 = np.array([-2,1])

x1 = np.array([1,2])

x2 = np.array([-3,-2])
fig, axs = plt.subplots(1, 2, figsize=(10,5))

fx0 = np.matmul(A,x0)

fx1 = np.matmul(A,x1)

fx2 = np.matmul(A,x2)

evec1 = np.matmul(A,vec1)

evec2 = np.matmul(A,vec2)



print("Before applying A: " , "x0 =",x0,"x1 =",x1,"x2 =",x2,"eigenvector1 =",vec1,"eigenvector2 =",vec2)

print("After applying A: " , "x0 =",fx0,"x1 =",fx1,"x2 =",fx2,"eigenvector1 =",evec1,"eigenvector2 =",evec2)

axs[0].quiver(x0[0],x0[1], angles='xy',scale_units='xy', scale = 1)

axs[0].quiver(x1[0],x1[1], angles='xy',scale_units='xy', scale = 1)

axs[0].quiver(x2[0],x2[1], angles='xy',scale_units='xy', scale = 1)

axs[0].quiver(vec1[0],vec1[1], angles='xy',scale_units='xy', scale = 1,color='red')

axs[0].quiver(vec2[0],vec2[1], angles='xy',scale_units='xy', scale = 1,color='red')

axs[1].quiver(fx0[0],fx0[1], angles='xy',scale_units='xy', scale = 1)

axs[1].quiver(fx1[0],fx1[1], angles='xy',scale_units='xy', scale = 1)

axs[1].quiver(fx2[0],fx2[1], angles='xy',scale_units='xy', scale = 1)

axs[1].quiver(evec1[0],evec1[1], angles='xy',scale_units='xy', scale = 1,color='red')

axs[1].quiver(evec2[0],evec2[1], angles='xy',scale_units='xy', scale = 1,color='red')

axs[0].set_xlim(-5,5)

axs[0].set_ylim(-5,5)

axs[0].grid()

axs[0].set_title("Vectors before A")

axs[1].set_title("Vectors after A")

plt.xlim(-5,5)

plt.ylim(-5,5)

plt.grid()

plt.show()
x = np.array([2,1])

A = np.array([[0,-1],

              [1,0]])

fx = np.matmul(A,x)

fig, axs = plt.subplots(1,2, figsize=(10,5))

axs[0].quiver(x[0],x[1],angles='xy',scale_units='xy', scale = 1)

axs[1].quiver(fx[0],fx[1],angles='xy',scale_units='xy', scale = 1)

axs[0].set_xlim(-5,5)

axs[0].set_ylim(-5,5)

axs[0].grid()

axs[0].set_title("Vectors before A")

axs[1].set_title("Vectors after A")

plt.xlim(-5,5)

plt.ylim(-5,5)

plt.grid()

plt.show()



print("Before A:", "x =",x)

print("After A:", "x =", fx)
x = np.array([1,1])

A = np.array([[(2**(0.5))/2,-(2**(0.5))/2],

              [(2**(0.5))/2,(2**(0.5))/2]])

fx = np.matmul(A,x)

fig, axs = plt.subplots(1,2, figsize=(10,5))

axs[0].quiver(x[0],x[1],angles='xy',scale_units='xy', scale = 1)

axs[1].quiver(fx[0],fx[1],angles='xy',scale_units='xy', scale = 1)

axs[0].set_xlim(-5,5)

axs[0].set_ylim(-5,5)

axs[0].grid()

axs[0].set_title("Vectors before A")

axs[1].set_title("Vectors after A")

plt.xlim(-5,5)

plt.ylim(-5,5)

plt.grid()

plt.show()



print("Before A:", "x =",x)

print("After A:", "x =", fx)
x = np.array([1,0])

pi = 3.141592653589793

theta = 3*pi/4

A = np.array([[np.cos(theta),-np.sin(theta)],

              [np.sin(theta),np.cos(theta)]])

fx = np.matmul(A,x)

fig, axs = plt.subplots(1,2, figsize=(10,5))

axs[0].quiver(x[0],x[1],angles='xy',scale_units='xy', scale = 1)

axs[1].quiver(fx[0],fx[1],angles='xy',scale_units='xy', scale = 1)

axs[0].set_xlim(-5,5)

axs[0].set_ylim(-5,5)

axs[0].grid()

axs[0].set_title("Vectors before A")

axs[1].set_title("Vectors after A")

plt.xlim(-5,5)

plt.ylim(-5,5)

plt.grid()

plt.show()



print("Before A:", "x =",x)

print("After A:", "x =", fx)