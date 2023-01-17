# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/fifa19/data.csv')
data.info()
data.head(10)  # Datasetteki ilk 10 oyuncuyu gösterir
data.columns  # Dataset sütun baişlıkları
# Scatter Plot 

# x = Finishing, y = LongShots

data.plot(kind='scatter', x='Finishing', y='LongShots',alpha = 0.5,color = 'red')

plt.xlabel('Finishing')                                    # label = name of label

plt.ylabel('LongShots')

plt.title('Finishing - LongShots Scatter Plot')            # title = title of plot
# Scatter Plot 

# x = Balance, y = BallControl

data.plot(kind='scatter', x='Balance', y='BallControl',alpha = 0.4,color = 'green')

plt.xlabel('Balance')                                    # label = name of label

plt.ylabel('BallControl')

plt.title('Balance - BallControl Scatter Plot')            # title = title of plot
data.Balance.plot(kind = 'hist',bins = 50,figsize = (10,10))

plt.show()
data.head(7)
# Player Dictionary Name & Overall



dict_player = {"L. Messi":94, "Cristiano Ronaldo":94, "Neymar Jr":92, "De Gea":91, "K. De Bruyne":91}

dict_player
# add new player

dict_player["E.Hazard"] = 91

dict_player
dict_player.keys()
dict_player.values()
for key,value in dict_player.items():

    print("Name:    ", key, "\nOverall: ",value)

print("  ")