# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import seaborn as sns

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_art = pd.read_csv('../input/data-set-1/public_art.csv', engine='python')
df = data_art

data_art.YearOfInstallation.groupby(df.YearOfInstallation.sub(1905)//20).mean()

df.groupby('YearOfInstallation').YearOfInstallation.count()

bins = [1900, 1920, 1940, 1960, 1980, 2000, 2020]

groups = df.groupby(pd.cut(df.YearOfInstallation, bins))

groups.YearOfInstallation.count()

DataFrame = groups.YearOfInstallation.count()

print(DataFrame)

from pandas import DataFrame



ArtByInstallation = {'YearOfInstallation': ['1900-1920','1920-1940','1940-1960','1960-1980','1980-2000','2000-2020'],

        'Count of Art Per 20 Years': [6,11,23,71,141,242] }



df = DataFrame(ArtByInstallation,columns= ['YearOfInstallation', 'Count of Art Per 20 Years'])

print (df)





height = [6, 11, 23, 71, 141, 242]

y_pos = np.arange(len(bars))

plt.bar(y_pos, height, color=('red', 'orange', 'yellow','green','skyblue','pink'))

plt.xlabel('Years', fontweight='bold', color = 'black', fontsize='12', horizontalalignment='center')

plt.ylabel('Art Count',fontweight='bold')

plt.show

Art_Bar = df.plot(kind='bar', x='YearOfInstallation', y='Count of Art Per 20 Years',alpha = 0.7, color='aqua')

plt.xlabel('Years', fontweight='bold', color = 'black', fontsize='12', horizontalalignment='center')

plt.ylabel('Art Count',fontweight='bold')

plt.title('Number of Public Art By Years',fontweight='bold',fontsize='18')

    

plt.show
plt.art1 = plt.scatter(data_art['Longitude'],data_art['Latitude'])

plt.title('Public Art Scatter Plot',fontweight='bold',fontsize='18',color='navy')

plt.ylabel('Latitude',color='navy')

plt.xlabel('Longitude',color='navy')
data_ws = pd.read_csv('../input/data-set-1/public_washrooms.csv')

data_ws.plot(kind='scatter', x='LONGITUDE', y='LATITUDE',alpha = 0.7, color = 'green')

plt.title('Public Washrooms Scatter Plot')

plt.ylabel('Latitude')

plt.xlabel('Longitude')

plt5 = plt.scatter(data_art['Longitude'],data_art['Latitude'], alpha = 1, color = 'green', marker = '+')

plt5 = plt.scatter(data_ws['LONGITUDE'],data_ws['LATITUDE'], alpha = 0.99, color = 'orange', marker = '>')

plt.ylabel('Latitude')

plt.xlabel('Longitude')

plt.legend(['Public Art','Public Washroom'])

plt.title('Public Art and Washroom Scatter Plot')

data_ws.info()

list(data_ws.columns)
data_ws.head(106)
data_ws2 = data_ws.drop(['NAME','ADDRESS','TYPE','LOCATION','SUMMER_HOURS','WINTER_HOURS','NOTE','LATITUDE','LONGITUDE','MAINTAINER'],axis=1)

data_ws3=data_ws2.drop([67,73,74], axis=0)

P = data_ws3.groupby('WHEELCHAIR_ACCESS')['PRIMARYIND'].count().reset_index()



P['Percentage'] = 100 * P['PRIMARYIND']  / P['PRIMARYIND'].sum()



print(P)
objects = ('YES', 'NO')

y_pos = np.arange(len(objects))

performance = [39,63]

 

plt.bar(y_pos, performance, align='center', alpha=0.8, color = 'purple')

plt.xticks(y_pos, objects)

plt.ylabel('Number of Public Washrooms',color = 'grey')

plt.title('Wheel Chair Accessibility',color='grey',fontsize='23')

 

plt.show()
ws_access = data_ws[data_ws['WHEELCHAIR_ACCESS']=='Yes']

ws_noaccess = data_ws[data_ws['WHEELCHAIR_ACCESS']=='No']

ws_access.head(5)

ws_noaccess.head(6)



plt2 = plt.scatter(ws_access['LONGITUDE'],ws_access['LATITUDE'],color='green',label='Accessible')

plt2 = plt.scatter(ws_noaccess['LONGITUDE'],ws_noaccess['LATITUDE'],color='red',label='Inaccessible')

plt.legend()

plt.title('Location of Public Washrooms Based On Accessibility')