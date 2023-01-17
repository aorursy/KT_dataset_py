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
import pandas as pd 

import numpy as np

import matplotlib as plt 

import seaborn as sns 
import socket

socket
dataset1 = pd.read_csv("../input/agricuture-crops-production-in-india/datafile (1).csv")

dataset2 = pd.read_csv("../input/agricuture-crops-production-in-india/datafile (2).csv")

dataset3 = pd.read_csv("../input/agricuture-crops-production-in-india/datafile (3).csv")

dataset4 = pd.read_csv("../input/agricuture-crops-production-in-india/datafile.csv")

dataset5 = pd.read_csv("../input/agricuture-crops-production-in-india/produce.csv")
dataset1.columns = ['Crop','State','CostofCultivationA2',

                     'CostofCultivationC2','CostofProduction2','Yield']
print(dataset1.shape)

print(dataset2.shape)

print(dataset3.shape)

print(dataset4.shape)

print(dataset5.shape)
dataset1.info()
#check ค่า missing values

print('============1=====================')

print(dataset1.isnull().sum())

print(dataset1.shape)

print('===============2==================')

print(dataset2.isnull().sum())

print(dataset2.shape)

print('==============3===================')

print(dataset3.isnull().sum())

print(dataset3.shape)

print('==================4===============')

print(dataset4.isnull().sum())

print(dataset4.shape)

print('=============5====================')

print(dataset5.isnull().sum())

print(dataset5.shape)
sns.pairplot(dataset1)
ax = sns.boxplot(x="Crop", y="CostofCultivationA2", data=dataset1, palette="Set3")
print('============1=====================')

print(dataset1.nunique())

print('===============2==================')

print(dataset2.nunique())

print('==============3===================')

print(dataset3.nunique())

print('==================4===============')

print(dataset4.nunique())

print('=============5====================')

print(dataset5.nunique())

dataset1.groupby('Crop').sum()


data = {'ARHAR': '1', 'COTTON':'2' ,'GRAM': '3', 'GROUNDNUT':'4','MAIZE': '5', 'MOONG':'6','PADDY': '7', 'RAPESEED AND MUSTARD':'8','SUGARCANE': '9', 'WHEAT':'10' }

dataset1['Crop'] = dataset1['Crop'].map(data)

dataset1.head()
sns.pairplot(dataset1 ,hue = 'Crop',diag_kind = 'kde',kind = "scatter",palette = "husl") 
dataset1.groupby('State').sum()
dataset1 = pd.read_csv("../input/agricuture-crops-production-in-india/datafile (1).csv")

dataset1.columns = ['Crop','State','CostofCultivationA2',

                     'CostofCultivationC2','CostofProduction2','Yield']

dataset1.head()
# sns.pointplot(x='CostofCultivationA2',y='Yield',data=dataset1,color='lime',alpha=0.8)

# sns.pointplot(x='CostofCultivationC2',y='Yield',data=dataset1,color='blue',alpha=0.8)

# sns.pointplot(x='CostofProduction2',y='Yield',data=dataset1,color='pink',alpha=0.8)

# plt.text(4,0.6,'CostofCultivationA2',color='lime',fontsize = 17,style = 'italic')

# plt.text(4,0.55,'CostofCultivationC2',color='blue',fontsize = 17,style = 'italic')

# plt.text(4,0.55,'CostofProduction2',color='pink',fontsize = 17,style = 'italic')

# plt.xlabel('Crop',fontsize = 15,color='blue')

# plt.ylabel('State',fontsize = 15,color='blue')

# plt.title('Heals vs Boosts',fontsize = 20,color='blue')

# plt.grid()

# plt.show()
data = {'ARHAR': '1', 'COTTON':'2' ,'GRAM': '3', 'GROUNDNUT':'4','MAIZE': '5', 'MOONG':'6','PADDY': '7', 'RAPESEED AND MUSTARD':'8','SUGARCANE': '9', 'WHEAT':'10' } 

dataset1['Crop'] = dataset1['Crop'].map(data)

dataset1 = dataset1.loc[dataset1.Crop == '2']

sns.pairplot(dataset1 ,hue = 'Crop',diag_kind = 'kde',kind = "scatter",palette = "husl")    

# dataset1.head()