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

import matplotlib.pyplot as plt
goals = [18,22,29,31,41,15,26,9,14,20]

season_number = np.arange(len(goals))

print(season_number) 
plt.plot(season_number,goals)

plt.xlabel("Season Number")

plt.ylabel("Goals")

plt.title("Daniel Sedins's goals over time")

plt.show()


ae_goals = [np.nan,np.nan,np.nan,1,8,10,5,8,11,8]

le_goals = [6,14,36,29,27,26,12,10,22,30] 

jm_goals = [np.nan,np.nan,0,6,23,np.nan,9,3,2,30]

tp_goals = [np.nan,np.nan,3,12,15,24,15,0,9,9]
plt.plot(season_number,ds_goals,label = "Daniel Sedin")

plt.plot(season_number,ae_goals,label = "Alexander Edler")

plt.plot(season_number,le_goals,label = "Loui Eriksson")

plt.plot(season_number,jm_goals,label = "Jacob Markstrom")

plt.plot(season_number,tp_goals,label = "Tanner Pearson")

plt.legend()



plt.xlabel("Season Number")

plt.ylabel("Goals")

plt.title("The Vancouver Canuck's total goals per season over time")

plt.show()
goals_std = np.std(goals)

print("Standard Deviation:", goals_std)
# What I've commented off below is what we plotted in Homework 2. Look how similar the code is!

# plt.plot(season_number,goals)

plt.errorbar(season_number,goals,

             yerr=goals_std)

plt.xlabel("Season Number")

plt.ylabel("Goals")

plt.title("Daniel Sedin's Goals over time")

# I also want to restrict the width of my y axis because negative points are not possible (well, maybe...)

# Use your discression here, but think about what limits might make sense in your case!

plt.ylim(0,55)

plt.show()
from scipy.optimize import curve_fit
def f(x, A, B): # this is your 'straight line' y=f(x)

    return A*x + B
popt, pcov = curve_fit(f, season_number, goals) # your data x, y to fit

print("Slope:",popt[0]) 

print("Intercept:",popt[1])
# y = m*x + b

y_fit = popt[0]*season_number + popt[1]
plt.errorbar(season_number,goals,

             yerr=goals_std)

# the fit!

plt.plot(season_number, y_fit,'--')

plt.xlabel("Season Number")

plt.ylabel("Goals")

plt.title("Daniel Sedin's Goals over time")

plt.ylim(0,55)

plt.show()
toi = [1111,1366,1463,1562,1541,1206,1521,1355,894,1504,1505]

season_number = np.arange(len(toi))

print(season_number) 
plt.plot(season_number,toi)

plt.xlabel("Season Number")

plt.ylabel("Time on Ice")

plt.title("Daniel Sedins's Time on Ice over 10 seasons")

plt.show()