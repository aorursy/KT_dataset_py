# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

x = np.array ([1, 2, 3, 4, 5])

s = (np.sum ([np.cos(x)**2])) * 0.5

print (s)
import numpy as np

w = np.array ([10,20,30,40,50])

x = np.array ([1, 2, 3, 4, 5])

b = const = 5

s = np.sum(np.dot(w,x)+b)

print (s)
import numpy as np

y = np.array ([10, 20, 30, 40, 50])

t = np.array ([1, 2, 3, 4, 5])

f = np.sum((y-t)**2)

print (f)
import matplotlib.pyplot as plt

import numpy as np

fig, ax = plt.subplots()

x = np.linspace(0.1, 10,100)

y = 1/x

ax.plot(x,y, color = '#0a0b0c')

fig.set_figwidth(10)

fig.set_figheight(3)

plt.show ()
import numpy as np

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

x = np.linspace(1, 10, 100)

i = np.linspace(1, 10, 100)

h = (x-1)/ 100

xi = 1+i*h

y = (np.sum(1/xi)) * h

ax.plot(xi,y, color = '#0a0b0c')

fig.set_figwidth(10)

fig.set_figheight(7)

plt.show ()
import matplotlib.pyplot as plt

import numpy as np

x = np.linspace (-6, 6, 100)

y = np.cos(x)

fig, ax = plt.subplots()

mw = 0

sig = 0.6

a = np.random.normal(mw, sig, 100)

b = np.random.normal(mw, sig, 100)

ax.scatter(a+x, b+y, c = 'r')

ax.plot(x,y)

fig.set_figwidth(15)

fig.set_figheight(10)

plt.show()
import matplotlib as plt

import numpy as np

N = const = 5

a = np.arange(2*N)

a = a + 1

a = np.random.permutation(a)

a = a.reshape(N, 2)



print(a)
import matplotlib.pyplot as plt

import numpy as np

f = np.linspace(0.2*np.pi, 1000)

a = 2

k = 2

Y = 0

fig, ax = plt.subplots()

r = a * np.cos(k * f + Y)

x =r * np.cos(f)

y = r * np.sin(f)



ax.plot(x,y, c = "r")

fig.set_figwidth(15)

fig.set_figheight(15)

plt.show()
import matplotlib.pyplot as plt

import numpy as np

x=np.array([[1,2], [2, 0.5], [6,1],[-2,1],[-4,0],[1,1],[2,-6],[-2,-4]])

y=np.array([[1],[1],[0],[0],[2],[1],[1],[2]])

Y=np.concatenate((x, y), axis=1)

Y, counts=(np.unique(y, return_counts=True))

print (counts)

 

import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(-1, 1, 100)

y = np.linspace (-1, 1, 100)

X,Y = np.meshgrid(x,y)

F = X**2 + Y**2 - 0.06

plt.contour ( X,Y,F,[0])

plt.show()