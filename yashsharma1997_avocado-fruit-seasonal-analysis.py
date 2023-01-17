# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
from plotly.offline import iplot
cf.go_offline()
# I am going to use following Visualisation libraries for a better UI
data=pd.read_csv('../input/avocado.csv',index_col=0)
data.head()
from datetime import datetime
#To make column Date readable by Pandas
data['Date']=pd.to_datetime(data['Date'])
del data['4046']
del data['4225']
del data['4770']
data.head()
#Skimming dataframe as per the need
def month(x):
    if (x==1):
        return 'Jan'
    elif (x==2):
        return 'Feb'
    elif (x==3):
        return 'Mar'
    elif (x==4):
        return 'Apr'
    elif (x==5):
        return 'May'
    elif (x==6):
        return 'Jun'
    elif (x==7):
        return 'Jul'
    elif (x==8):
        return 'Aug'
    elif (x==9):
        return 'Sept'
    elif (x==10):
        return 'Oct'
    elif (x==11):
        return 'Nov'
    else:
        return 'Dec'
def season(x):
    if(x=='Mar' or x=='Apr' or x=='May'):
        return 'Spring'
    elif(x=='Jun' or x=='Jul' or x=='Aug'):
        return 'Summer'
    elif(x=='Sept' or x=='Oct' or x=='Nov'):
        return 'Autumn'
    else:
        return 'Winter'
data['Month']=data['Date'].dt.month.apply(month)
data['Season']=data['Month'].apply(season)
#To make it more readable :)
sns.boxplot('Month','AveragePrice',data=data,hue='Season')
plt.tight_layout(rect=(0,0,2,2))
k=pd.DataFrame(data.groupby('region').mean()['AveragePrice'].sort_values())
#to find average prices as per Region.
k.iplot(kind='scatter')
data['Revenue']=data['AveragePrice']*data['Total Volume']
data.head()
rev=data.groupby('Month').sum()['Revenue']
rev.iplot()