# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #data visualization
import matplotlib.pyplot as plt #plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#creating an dataframe
df = pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_16.csv')

data = df[['short_name','age','height_cm','weight_kg','nationality','club',
'overall','potential',
 'player_positions',
 'preferred_foot',
 'international_reputation',
 'weak_foot',
 'skill_moves',
 'body_type',
 'team_position',
 'pace',
 'shooting',
 'passing',
 'dribbling',
 'defending',
 'physic']]
data.head(10) #looking first 10 index in dataframe
data.info()
data.corr() #the process of establishing a relationship or connection between features.
data.describe()
#Correlation map
f,ax = plt.subplots(figsize = (15,15))
sns.heatmap(data.corr(),annot = True,linewidths=.5,fmt = '.1f',ax=ax, cmap="BuPu")
plt.show()
data.columns
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.overall.plot(kind = 'line',color = 'red',label = 'overall',linewidth = 1,alpha = 0.8, grid = True,linestyle = ':')
data.potential.plot(color = 'green',label='potential',linewidth = 1,alpha = 0.8,grid=True,linestyle = '-.')
plt.legend(loc = 'upper right')    # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
# Scatter Plot 
# x = Physic, y = Defending
data.plot(kind='scatter', x='physic', y='defending',alpha = 0.5,color = 'red')
plt.xlabel('physic')            
plt.ylabel('defence')
plt.title('Physic Defense Scatter Plot')           
# Histogram
# bins = number of bar in figure
data.dribbling.plot(kind = 'hist',bins = 80,figsize = (12,12))
plt.xlabel('Dribbling')
plt.show()
# clf() = cleans it up again you can start a fresh
data.dribbling.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()
dictionary = {'Spain': 'Barcelona' , 'England':'Liverpool', 'Turkey':'Galatasaray'}
print(dictionary.keys())
print(dictionary.values())
dictionary['Spain'] = 'Real Madrid' # We updated the existing entry
print(dictionary)
dictionary['Holland'] = 'Ajax' # We added a new entry
print(dictionary)
del dictionary['England'] # We deleted the key and value from dictionary
print(dictionary)
print('Holland' in dictionary) # If dictionary has the key named Holland, it will make turn True.
dictionary.clear()                   # remove all entries in dictionary
print(dictionary)


df = pd.read_csv('../input/fifa-20-complete-player-dataset/players_20.csv')
data = df[['short_name','age','height_cm','weight_kg','nationality','club',
'overall','potential',
 'player_positions',
 'preferred_foot',
 'international_reputation',
 'weak_foot',
 'skill_moves',
 'body_type',
 'team_position',
 'pace',
 'shooting',
 'passing',
 'dribbling',
 'defending',
 'physic']]
series = data['defending'] # it gives us the parameters as vector that we wrote 
print(type(series))
data_frame = data[['defending']] # it gives us the parameters as data frame that we wrote 
print(type(data_frame))
# Comparison operator
print(5 > 2)
print(5!=2)
# Boolean operators
print(True and False)
print(True or False)
# 1 - Filtering Pandas data frame
x = data['dribbling']>90 # it gives us the shooting values which  bigger than 85 
data[x] # It prints the True values as table
# 2 - Filtering pandas with logical_and
# We will see young stars.
data[(data['age']<25) & (data['overall']>85)]
