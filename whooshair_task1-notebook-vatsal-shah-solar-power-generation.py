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
import math
goat2= pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
goat3= pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
goat4= pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
goat5= pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
goat2.nunique()
goat3.nunique()
goat4.nunique()
goat5.nunique()
goat2.describe()

goat3.describe()

goat4.describe()

goat5.describe()

goat5.columns

goat2.columns

goat3.columns

goat4.columns

dy1_mean = goat2['DAILY_YIELD'].mean()
print(dy1_mean)
dy2_mean = goat4['DAILY_YIELD'].mean()
print(dy2_mean)
goat4

goat3['DATE_TIME'] = pd.to_datetime(goat3['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')

goat3['DATE'] = goat3['DATE_TIME'].apply(lambda x:x.date())
goat3['TIME'] = goat3['DATE_TIME'].apply(lambda x:x.time())

goat3['HOUR'] = pd.to_datetime(goat3['TIME'],format='%H:%M:%S').dt.hour
goat3['MINUTES'] = pd.to_datetime(goat3['TIME'],format='%H:%M:%S').dt.minute

goat2['DATE_TIME'] = pd.to_datetime(goat2['DATE_TIME'],format = '%d-%m-%Y %H:%M')

goat2['DATE'] = goat2['DATE_TIME'].apply(lambda x:x.date())
goat2['TIME'] = goat2['DATE_TIME'].apply(lambda x:x.time())

goat2['HOUR'] = pd.to_datetime(goat2['TIME'],format='%H:%M:%S').dt.hour
goat2['MINUTES'] = pd.to_datetime(goat2['TIME'],format='%H:%M:%S').dt.minute



goat5['DATE_TIME'] = pd.to_datetime(goat5['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')

goat5['DATE'] = goat5['DATE_TIME'].apply(lambda x:x.date())
goat5['TIME'] = goat5['DATE_TIME'].apply(lambda x:x.time())

goat5['HOUR'] = pd.to_datetime(goat5['TIME'],format='%H:%M:%S').dt.hour
goat5['MINUTES'] = pd.to_datetime(goat5['TIME'],format='%H:%M:%S').dt.minute

goat4['DATE_TIME'] = pd.to_datetime(goat4['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')

goat4['DATE'] = goat4['DATE_TIME'].apply(lambda x:x.date())
goat4['TIME'] = goat4['DATE_TIME'].apply(lambda x:x.time())

goat4['HOUR'] = pd.to_datetime(goat4['TIME'],format='%H:%M:%S').dt.hour
goat4['MINUTES'] = pd.to_datetime(goat4['TIME'],format='%H:%M:%S').dt.minute
goat3.groupby('DATE')['IRRADIATION'].sum()
maxambip1= goat3['AMBIENT_TEMPERATURE'].max()
print(maxambip1)
maxambip2= goat5['AMBIENT_TEMPERATURE'].max()
print(maxambip2)
maxmodtp1= goat3['MODULE_TEMPERATURE'].max()
print(maxmodtp1)
maxmodtp2= goat5['MODULE_TEMPERATURE'].max()
print(maxmodtp2)
noip1= goat2['SOURCE_KEY'].nunique()
print(noip1) 
noip2= goat4['SOURCE_KEY'].nunique()
print(noip2) 
maxdcpp1= goat2['DC_POWER'].max()
print(maxdcpp1)
maxacpp1= goat2['AC_POWER'].max()
print(maxacpp1)
maxdcpp2= goat4['DC_POWER'].max()
print(maxdcpp2)
maxacpp2= goat4['AC_POWER'].max()
print(maxacpp2)
mindcpp1= goat2['DC_POWER'].min()
print(mindcpp1)
minacpp1= goat2['AC_POWER'].min()
print(minacpp1)
mindcpp2= goat4['DC_POWER'].min()
print(mindcpp2)
minacpp2= goat4['AC_POWER'].min()
print(minacpp2)
goat2.groupby('DATE')[['AC_POWER','DC_POWER']].sum().max()

goat2.groupby('DATE')[['AC_POWER','DC_POWER']].sum().min()

goat4.groupby('DATE')[['AC_POWER','DC_POWER']].sum().max()

goat4.groupby('DATE')[['AC_POWER','DC_POWER']].sum().min()
maxdcinvp1 = goat2[goat2['DC_POWER']== goat2["DC_POWER"].max()]['SOURCE_KEY']
print(maxdcinvp1)

maxacinvp1 = goat2[goat2['AC_POWER']== goat2["AC_POWER"].max()]['SOURCE_KEY']
print(maxacinvp1)

maxdcinvp2 = goat4[goat4['DC_POWER']== goat4["DC_POWER"].max()]['SOURCE_KEY']
print(maxdcinvp2)

maxacinvp2 = goat4[goat4['AC_POWER']== goat4["AC_POWER"].max()]['SOURCE_KEY']
print(maxacinvp2)

def takeSecond(elem):
    return elem[1]

inv = goat2['SOURCE_KEY'].unique()
ac_inv = goat2.groupby('SOURCE_KEY')['AC_POWER'].mean()

p_list=[]

for i in range(len(inv)):
     p_list.append([inv[i], ac_inv[i]])
    
p_list.sort(reverse = True ,key=takeSecond)

for i in range(len(p_list)):
    print("RANK", i+1, ":- ##",p_list[i][0], "## AVERAGE AC OUPUT IS:-", p_list[i][1], "\n" )
date_unique= goat2['DATE'].value_counts()
date_unique
24*4*22