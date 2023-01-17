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
#Read
df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_pgen3 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df_pgen4 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
df_pgen1.head()#generator 1
df_pgen2.head()#weather sensor 1
df_pgen2.tail()
df_pgen3.head()#generator 2
df_pgen4.head()#weather sensor 2
#df_pgen1['DATE_TIME'].head()
#change all object into datetime dtypes
df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')
df_pgen2['DATE_TIME'] = pd.to_datetime(df_pgen2['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df_pgen3['DATE_TIME'] = pd.to_datetime(df_pgen3['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df_pgen4['DATE_TIME'] = pd.to_datetime(df_pgen4['DATE_TIME'],format = '%Y-%m-%d %H:%M')
#df_pgen1.dtypes
#mean value of daily yield from generator 1
print("Mean Daily Yield from generator 1 : {}".format(df_pgen1['DAILY_YIELD'].mean()))
#mean value of daily yield from generator 2
print("Mean Daily Yield from generator 2 : {}".format(df_pgen3['DAILY_YIELD'].mean()))



#finding the mean of both generator 

total_dailyyield = df_pgen1['DAILY_YIELD'].sum()+df_pgen3['DAILY_YIELD'].sum()
total_generator = len(df_pgen3['DAILY_YIELD'])+len(df_pgen1['DAILY_YIELD'])

Mean_of_Both = float(total_dailyyield)/float(total_generator)
print("Mean from both generator : {}".format( Mean_of_Both ))
#df_pgen2.tail()
#df_pgen4.tail() 

#just taking the date 
df_pgen2['DATE'] = pd.to_datetime(df_pgen2['DATE_TIME']).dt.date
df_pgen2['DATE'] = pd.to_datetime(df_pgen2['DATE'])


df_pgen4['DATE'] = pd.to_datetime(df_pgen4['DATE_TIME']).dt.date
df_pgen4['DATE'] = pd.to_datetime(df_pgen4['DATE'])

#sum by day
print('Total Irridation per day in Generator 1:')
Irridation1 = df_pgen2.groupby([df_pgen2['DATE'].dt.date])['IRRADIATION'].sum()
print(Irridation1)

print("\n\n")

print('Total Irridation per day in Generator 2:')
Irridation2 = df_pgen4.groupby([df_pgen4['DATE'].dt.date])['IRRADIATION'].sum()
print(Irridation2)
print("Generator 1")
print("Max Ambient : {}".format(df_pgen2['AMBIENT_TEMPERATURE'].max()))
print("Max Module Temperature : {}\n\n".format(df_pgen2['MODULE_TEMPERATURE'].max()))

print("Generator 2")
print("Max Ambient : {}".format(df_pgen4['AMBIENT_TEMPERATURE'].max()))
print("Max Module Temperature : {}".format(df_pgen4['MODULE_TEMPERATURE'].max()))
print("Generator 1 inverter  = {}".format(df_pgen1['SOURCE_KEY'].nunique()))#the inverter only available in the solar
print("Generator 2 inverter  = {}".format(df_pgen3['SOURCE_KEY'].nunique()))#power generator,(1 & 3 == generator)
#copy paste from above 
#just taking the date for dataframe 1 for Generator 1
df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE_TIME']).dt.date
df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'])

#max & min AC Power by day
print('Maximum AC power in Generator 1 :')
Max_AC_1 = df_pgen1.groupby([df_pgen1['DATE'].dt.date])['AC_POWER'].max()
print(Max_AC_1)

print('\n\n\nMinimum AC Power in Generator 1 :')
Min_AC_1 = df_pgen1.groupby([df_pgen1['DATE'].dt.date])['AC_POWER'].min()
print(Min_AC_1)

#max & min DC Power by day
print('Maximum DC power in Generator 1 :')
Max_DC_1 = df_pgen1.groupby([df_pgen1['DATE'].dt.date])['DC_POWER'].max()
print(Max_DC_1)

print('\n\n\nMinimum DC Power in Generator 1 :')
Min_DC_1 = df_pgen1.groupby([df_pgen1['DATE'].dt.date])['DC_POWER'].min()
print(Min_DC_1)

#copy paste from above 
#just taking the date for dataframe 3 for Generator 2
df_pgen3['DATE'] = pd.to_datetime(df_pgen3['DATE_TIME']).dt.date
df_pgen3['DATE'] = pd.to_datetime(df_pgen3['DATE'])

#max & min AC Power by day
print('Maximum AC power in Generator 3 :')
Max_AC_2 = df_pgen3.groupby([df_pgen1['DATE'].dt.date])['AC_POWER'].max()
print(Max_AC_2)

print('\n\n\nMinimum AC Power in Generator 3 :')
Min_AC_2 = df_pgen3.groupby([df_pgen1['DATE'].dt.date])['AC_POWER'].min()
print(Min_AC_2)

#max & min DC Power by day
print('Maximum DC power in Generator 3 :')
Max_DC_2 = df_pgen3.groupby([df_pgen1['DATE'].dt.date])['DC_POWER'].max()
print(Max_DC_2)

print('\n\n\nMinimum DC Power in Generator 3 :')
Min_DC_2 = df_pgen3.groupby([df_pgen1['DATE'].dt.date])['DC_POWER'].min()
print(Min_DC_2)
print("Generator 1")
print(df_pgen1[df_pgen1['DC_POWER'] == df_pgen1['DC_POWER'].max()]['SOURCE_KEY'],"\n")#max DC
print(df_pgen1[df_pgen1['AC_POWER'] == df_pgen1['AC_POWER'].max()]['SOURCE_KEY'])#max AC

print("\n\ngenerator 2")
print(df_pgen3[df_pgen3['DC_POWER'] == df_pgen3['DC_POWER'].max()]['SOURCE_KEY'],"\n")#max DC
print(df_pgen3[df_pgen3['AC_POWER'] == df_pgen3['AC_POWER'].max()]['SOURCE_KEY'])#max AC
print("Inverter With The Maximum AC/DC Power Rank in Generator 1")
'''
print(df_pgen1.sort_values(by='DC_POWER', ascending=False)['SOURCE_KEY'].unique())#the source key indicate the inverter
print("AC")

print(df_pgen1.sort_values(by='AC_POWER', ascending=False)['SOURCE_KEY'].unique())
print(df_pgen1.sort_values(by='AC_POWER', ascending=False))#full information
'''
inverter_1 =df_pgen1.sort_values(by='DC_POWER', ascending=False)['SOURCE_KEY'].unique()#since the inverter for maximum AC and DC same
for i in range(len(inverter_1)):
    print(i+1," ",inverter_1[i])

print("\nInverter With The Maximum AC/DC Power Rank in Generator 2")
inverter_2 =df_pgen3.sort_values(by='DC_POWER', ascending=False)['SOURCE_KEY'].unique()#since the inverter for maximum AC and DC same
for i in range(len(inverter_2)):
    print(i+1," ",inverter_2[i])


print("The sum of missing data in dataframe 1")
print(df_pgen1.isnull().sum())

print("\nThe sum of missing data in dataframe 2")
print(df_pgen2.isnull().sum())

print("\nThe sum of missing data in dataframe 3")
print(df_pgen3.isnull().sum())

print("\nThe sum of missing data in dataframe 4")
print(df_pgen4.isnull().sum())