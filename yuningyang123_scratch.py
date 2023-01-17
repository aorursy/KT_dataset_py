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

pts = [14,32,21,17,27,15,33,39,20,22,28]

print("number of games: ", len(pts)) # len() is the length of the list

print("range: ", np.min(pts), "-", np.max(pts))

print("points per game: ", np.mean(pts))

print("median: ", np.median(pts))

from statistics import mode #import the function 'mode'

try:

    print("mode: ", mode(pts))

except:

    print("there is no unique mode")

plt.hist(pts,6) # Even though the homework says 6 bins, I've put 10 to see the histogram better

plt.xlabel("points")

plt.ylabel("N")

plt.title("Kyrie Irving's points distribution")

plt.show()