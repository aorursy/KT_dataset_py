# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





vgs = pd.read_csv('../input/videogamesales/vgsales.csv')







print(vgs.describe())

vgs.info()

#LATEST YEAR IS 2015





#ALL TIME #1 IN SALES = PRINTS WII SPORTS

m1 = vgs.Global_Sales.max()

print(vgs.loc[vgs['Global_Sales'] == m1], 'Name]')









#WORK WITH DATA THAT IS FROM YEAR 2000 AND ON

vgs2000 = vgs.query('Year >= 2000')

print(vgs2000)







#PICKS THE TOP 100 GAMES

vgs100 = vgs.query('Rank < 101')

print(vgs100)







#PICKS THE TOP 100 GAMES FROM YEAR 2000 AND ON

vg = vgs.query('Year >= 2000 & Rank < 101')

print(vg)







#PLATFORM BREAKDOWN

print(vg['Platform'].value_counts())

print(vg.loc[vg['Platform'] == 'X360', 'Name'])

print(vg.loc[vg['Platform'] == 'Wii', 'Name'])

labels = 'X360', 'Wii', 'DS', 'PS3', '3DS', 'Others'

sizes = [16,15,13,9,7, 16]

plt.pie(sizes, labels=labels,shadow = True, startangle=90)

plt.show()







#GLOBAL SALES VS YEAR DOT GRAPH

vg.plot(kind='scatter', x = 'Year', y = 'Global_Sales', color = 'blue')

plt.xlabel('Year')

plt.ylabel('Global Sales')

plt.title('Global Sales VS Year')

plt.show()







#GENRE BREAKDOWN

print(vg['Genre'].value_counts())

labels = 'Shooter', 'Action', 'Role-Playing', 'Misc', 'Platform', 'Sports', 'Others'

sizes = [20,11,10,9,7,6,13]

explode = (0.1,0.1,0,0,0,0,0)

plt.pie(sizes, labels=labels,shadow = True, startangle=90, explode = explode, autopct='%1.1f%%')

plt.show()  









#PUBLISHERS

print(vg['Publisher'].value_counts())

labels = 'Nintendo', 'Activision', 'Take-Two Interactive', 'Microsoft Game Studios', 'Electronic Arts ', 'Others'

sizes = [34,14,9,6,5, 8]

plt.pie(sizes, labels=labels,shadow = True, startangle=90,autopct='%1.1f%%')

plt.show()





#BOXPLOT OF YEAR

a = sns.boxplot(vg.Year)

a.set(label = 'Year Rating')

a.set_title('Average Year Distribution')









print(np.percentile(vg.Global_Sales,50))

print(np.median(vg.Year))

print(np.mean(vg.Year))



#%%

vg = vgs.query('Year >= 2000 & Rank < 101')




