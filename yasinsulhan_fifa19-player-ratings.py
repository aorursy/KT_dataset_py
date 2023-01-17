# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#loading data

df = pd.read_csv("/kaggle/input/fifa19/data.csv")

df.head()

df.tail(5)
# as understood player photo is unnecessary for evaluating the data

df.drop(["Photo"],axis=1)
# information about data

df.info()

df.columns

df.describe().T
# relation between variables

df.corr()



#correlation map

f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(df.corr(),annot=True,linewidths=.10,fmt='.1f',ax=ax)
#Graphs



# the graph relation between speed and dribbling

df.Overall.plot(kind="line",color="blue",label="overral",linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df.Dribbling.plot(color="red",label="dribbling",linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()

# as you can see overall is directly proportional to dribbling





#scatter graph

# Scatter Plot 

# x =acceleration, y =sprint speed

df.plot(kind='scatter', x='Acceleration', y='SprintSpeed',alpha = 0.5,color = 'red')

plt.xlabel('Acceleration')              # label = name of label

plt.ylabel('SprintSpeed')

plt.title('Acceleration SprintSpeed Scatter Plot')



#histogram

df.Age.plot(kind="hist",bins=45)

plt.show()

### Filtering ###

a = df["Dribbling"]>90 #there are 12 player who has higher than 90 overall

df[a]



#the other way to filter player

df[np.logical_and(df["Dribbling"]>90 , df["SprintSpeed"]>90)] #there is only one player who provide such a condition



# by the way there is one more way to filter

df[(df['Dribbling']>90) & (df['SprintSpeed']>90)]