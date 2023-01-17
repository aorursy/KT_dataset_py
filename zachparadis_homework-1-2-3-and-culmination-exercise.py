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



#Kirk Cousins

yards = [98,230,174,233,306,333,338,285,220,220,319,276,243,207,122, 234]



print("number of games: ", len(yards))



print("range: ", np.min(yards),"-", np.max(yards))



print("yards per game: ", np.mean(yards))



print("median: ", np.median(yards))



from statistics import mode

print("mode: ", mode(yards))



plt.hist(yards,6)

plt.xlabel("Yards")

plt.ylabel("N")

plt.title("Kirk Cousins Yards Distribution")

plt.show
gamenumber = np.arange(len(yards))

print(gamenumber)

plt.plot(gamenumber,yards)

plt.xlabel("Game Number")

plt.ylabel("Yards")

plt.title("Kirk Cousins yards over time")

plt.show()
wilson_yards = [195, 300, 406, 240, 268, 295, 241, 182, 378, 232, 200, 240, 245, 286, 169, 233]

brady_yards  = [341, 264, 306, 150, 348, 334, 249, 259, 285, 216, 190, 326, 169, 128, 271, 221]

rodgers_yards= [203, 209, 235, 422, 238, 283, 429, 305, 161, 233, 104, 243, 195, 203, 216, 323]

rivers_yards = [333, 293, 318, 310, 211, 320, 329, 201, 294, 207, 353, 265, 314, 307, 279, 281]



plt.plot(gamenumber,yards,label = "Cousins")

plt.plot(gamenumber,wilson_yards,label = "Wilson")

plt.plot(gamenumber,brady_yards,label = "Brady")

plt.plot(gamenumber,rodgers_yards,label = "Rodgers")

plt.plot(gamenumber,rivers_yards,label = "Rivers")

plt.legend()



plt.xlabel("Game Number")

plt.ylabel("Yards")

plt.title("NFL Qb's yards over time")

plt.show()
yards_std = np.std(yards)

print("Standard Deviation:", yards_std)
plt.errorbar(gamenumber,yards,

             yerr=yards_std)

plt.xlabel("Game Number")

plt.ylabel("Yards")

plt.title("Kirk Cousins yards over time over time")



plt.ylim(0,500)

plt.show()
from scipy.optimize import curve_fit

def f(x, A, B): # this is your 'straight line' y=f(x)

    return A*x + B

popt, pcov = curve_fit(f, gamenumber, yards) # your data x, y to fit

print("Slope:",popt[0]) 

print("Intercept:",popt[1])

y_fit = popt[0]*gamenumber + popt[1]
plt.errorbar(gamenumber,yards,

             yerr=yards_std)



plt.plot(gamenumber, y_fit,'--')

plt.xlabel("Game Number")

plt.ylabel("Yards")

plt.title("Kirk Cousin's yards over time")

plt.ylim(0,350)

plt.show()