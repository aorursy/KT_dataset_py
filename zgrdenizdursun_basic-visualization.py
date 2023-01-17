# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/youtube-new/DEvideos.csv')

data.info()
data.columns
#correlation

plt.figure(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f')

plt.show()
#LinePlot between Likes & Dislikes

plt.figure(figsize=(15,15))

data.likes.plot(kind='line',color='orange',alpha=1,grid=True,linewidth=2,linestyle=':',label='Likes')

data.dislikes.plot(kind='line',color='black',alpha=1,grid=True,linewidth=2,linestyle='-.',label='Dislikes')

plt.legend(loc='upper right')     

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Likes & Dislikes Line Plot')          

plt.show()
#ScatterPlot between Likes & Dislikes

data.plot(kind='scatter',x='likes',y='dislikes',color='orange',grid=True,alpha=.5)

plt.xlabel('Likes')              # label = name of label

plt.ylabel('Dislikes')

plt.title('Likes & Dislikes Scatter Plot')     

plt.show() #in order not the show description
#ScatterPlot between Likes & Dislikes

data.plot(kind='scatter',x='likes',y='comment_count',color='orange',grid=True,alpha=.5)

plt.xlabel('Likes')              # label = name of label

plt.ylabel('Comment Count')

plt.title('Likes & Comment Count Scatter Plot')     

plt.show() #in order not the show description
#likes histogram

restriction = data[data.likes < 2000] #Restriction uses for better visualization

restriction.likes.plot(kind = 'hist',bins = 50,figsize = (12,12),color='orange')

plt.show()

#dislikes histogram

restriction = data[data.dislikes < 2000]

restriction.dislikes.plot(kind = 'hist',bins = 50,figsize = (12,12),color='black')

plt.show()

#comment count histogram

restriction = data[data.comment_count < 2000]

restriction.comment_count.plot(kind = 'hist',bins = 50,figsize = (12,12),color='green')

plt.show()