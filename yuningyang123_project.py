# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mpl_toolkits.mplot3d import Axes3D





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dis = 103.5 #meters distance from the kicking spot to the goal line

g = 9.81 #m/s^2

h = 2.4 #height of the goal, in meters

ball_radius = 0.11 #meters



N = 500 #Want a lot of points. I made a point for every half angle between 30 and 90 degrees

theta = np.linspace(30,60,N)

speed = np.linspace(22,28,N)



def v(dis,g,height,theta):

    r = theta * np.pi / 180

    return (dis/np.cos(r))*np.sqrt(g/(2*(np.tan(r)*dis - height)))



import matplotlib.pyplot as plt

plt.plot(theta,v(dis,g,h/2,theta),label = 'perfect')

plt.plot(theta,v(dis,g,0,theta),label = 'min')

plt.plot(theta,v(dis,g,h-ball_radius,theta),label = 'max')



plt.ylim(31,36)

plt.xlim(30,60)

#Label your own axes with units



plt.legend()

plt.xlabel("Angles (in degree)")

plt.ylabel("Kick Speed(in m/s)")

plt.title("Y Axis Angles vs Speed")

plt.grid() #to better see the plot

plt.show()

length = 7.32 #meters



theta_x_lim = np.arctan(length/dis/2) * 180 / np.pi #the range of x axis angles that is possible for a goal

theta_x = np.linspace(0, theta_x_lim, N)



def v_xy(dis,g,height,theta,theta_x):

    r = theta * np.pi / 180

    r_x = theta_x * np.pi / 180

    dis = dis / np.cos(r_x)

    return (dis/np.cos(r))*np.sqrt(g/(2*(np.tan(r)*dis - height)))



import matplotlib.pyplot as plt

X, Y = np.meshgrid(theta, theta_x)

Z_perfect = v_xy(dis,g,h/2,X,Y)

Z_min = v_xy(dis,g,0,X,Y)

Z_max = v_xy(dis,g,h-ball_radius,X,Y)

ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z_perfect, label='perfect')

ax.plot_surface(X, Y, Z_min, label='min')

ax.plot_surface(X, Y, Z_max, label='max')





ax.set_xlim(30, 60)

ax.set_ylim(0, 2.5)

ax.set_zlim(31, 36)

ax.set_xlabel("Y angle")

ax.set_ylabel("X angle")

ax.set_zlabel("Speed")

ax.dist = 10.5



plt.title("X, Y Angles vs Speed")

plt.show()
theta = np.linspace(30,68,N)

def v_xy_bounce(dis,g,height,theta, velocity):

    theta_result = []

    v_result = []

    for v in velocity:

        for t in theta:

            r = t * np.pi / 180

            d1 = 2 * np.sin(r) * np.cos(r) * v * v / g

            d2 = dis - d1

            t2 = d2 / 0.8 / v / np.cos(r)

            temp = 0.8 * v * np.sin(r) * 0.8 * v * np.sin(r) / 9.8 / 2 - 1/2 * g * (t2 - np.sin(r) * 0.8 * v / 9.8)

            if temp <= height + 0.03 and temp >= height - 0.03:

                theta_result.append(t)

                v_result.append(v)

    return np.array(theta_result), np.array(v_result)



import matplotlib.pyplot as plt

x_perfect, y_perfect = v_xy_bounce(dis,g,h/2,theta,speed)

x_min, y_min = v_xy_bounce(dis,g,0,theta,speed)

x_max, y_max = v_xy_bounce(dis,g,h-ball_radius,theta,speed)

plt.scatter(x_perfect, y_perfect, alpha=0.2, s=10, label='perfect')

plt.scatter(x_min, y_min, alpha=0.2, s=10, label='min')

plt.scatter(x_max, y_max, alpha=0.2, s=10, label='max')





plt.ylim(23,28)

plt.xlim(30,68)

#Label your own axes with units



plt.legend()

plt.xlabel("Angles (in degree)")

plt.ylabel("Kick Speed(in m/s)")

plt.title("Y Axis Angles vs Speed")

plt.grid() #to better see the plot

plt.show()




