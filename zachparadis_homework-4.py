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
N = 121 

theta = np.linspace(30,90,N)



def v(theta,distance,gravity,heightPerson,heightRim):

    x = distance

    g = gravity

    h = heightPerson

    H = heightRim

    radians = theta * np.pi / 180 #np.cos/tan take radian values

    return (x/np.cos(radians))*np.sqrt((g/2)/(x*np.tan(radians)+h-H))



distance = 15 #ft Distance to hoop

gravity = 32.174 #ft/s^2

heightPerson = 6 #ft Height of person

heightRim = 10 #ft Height of hoop

rim_radius = 0.738 #ft

ball_radius = 0.398 #ft



import matplotlib.pyplot as plt



plt.plot(theta,v(theta,distance-rim_radius+ball_radius,gravity,heightPerson,heightRim),label = 'min')

plt.plot(theta,v(theta,distance,gravity,heightPerson,heightRim),label = 'perfect')

plt.plot(theta,v(theta,distance+rim_radius-ball_radius,gravity,heightPerson,heightRim),label = 'max')







plt.ylim(24.3,27)

plt.xlim(38,68)



plt.xlabel("Angle (Degrees)")

plt.ylabel("Velocity (ft/s)")



plt.legend()

plt.grid() #to better see the plot

plt.show()