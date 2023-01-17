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



        

import matplotlib.pyplot as plt

import seaborn as sns

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filepath= '/kaggle/input/world-university-rankings/cwurData.csv'



data = pd.read_csv(filepath)

data.head()
data.corr()
f,ax = plt.subplots(figsize = (15,15))

sns.heatmap(data.corr(), annot = True, linewidth = .5, fmt = '.2f', ax=ax)

plt.show()
# Scatter Plot

data.plot(kind = 'scatter', x = 'broad_impact', y = 'world_rank',color = 'b', alpha = 0.5)

plt.xlabel('broad_impact')

plt.ylabel('world_rank')

plt.title('broad_impact  world_rank scatter plot ')

plt.show()
# Line plotting

##To investigate the data in 2015 we're filtering the data as it contains only 2015.

data = data[data['year'] == 2015]

data.broad_impact.plot(kind = 'line', color = 'g', label = 'broad_impact', linewidth = 1, alpha = 0.6, grid = True, linestyle = ':')

data.world_rank.plot(kind = 'line', color = 'blue', label = 'world_rank', linewidth = 1, alpha = 0.6, grid = True, linestyle = '-.')

plt.legend()

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
#As you can see in the scatter plot, universities broad impact increas cause to increase of their world record's.

#Histogram    To see the frequency of one column's variables.



data.publications.plot(kind = 'hist', bins = 100, figsize = (12,12),color = 'b',alpha = 0.8)

plt.show()
dF2015=data[data.year==2015]
##We commpare the scores of the unique countries 



dFUSA = dF2015[dF2015.country == 'USA']

dFUK = dF2015[dF2015.country == 'United Kingdom']

dFJAPAN = dF2015[dF2015.country == 'Japan']

dFSWITZERLAND = dF2015[dF2015.country == 'Switzerland']

dFISRAEL = dF2015[dF2015.country == 'Israel']
score_means =[dFUSA[0:5].score.mean(),dFUK[0:5].score.mean(),dFJAPAN[0:5].score.mean(),dFSWITZERLAND[0:5].score.mean(),dFISRAEL[0:5].score.mean()]
plt.bar(dF2015.country.unique()[0:5] , score_means[0:5], width=0.5, color='r',alpha = 0.7)

plt.show()