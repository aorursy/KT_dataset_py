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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_wgen1=pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df_wgen2=pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')
df_wgen1['DATE_TIME'] = pd.to_datetime(df_wgen1['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df_pgen1['DATE'] = df_pgen1['DATE_TIME'].apply(lambda x:x.date())
df_pgen1['TIME'] = df_pgen1['DATE_TIME'].apply(lambda x:x.time())

df_wgen1['DATE'] = df_wgen1['DATE_TIME'].apply(lambda x:x.date())
df_wgen1['TIME'] = df_wgen1['DATE_TIME'].apply(lambda x:x.time())

df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')
df_wgen1['DATE'] = pd.to_datetime(df_wgen1['DATE'],format = '%Y-%m-%d')

df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour
df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute

df_wgen1['HOUR'] = pd.to_datetime(df_wgen1['TIME'],format='%H:%M:%S').dt.hour
df_wgen1['MINUTES'] = pd.to_datetime(df_wgen1['TIME'],format='%H:%M:%S').dt.minute

_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen1.DATE_TIME,
        df_wgen1.AMBIENT_TEMPERATURE.rolling(window=20).mean(),
        label='Ambient',
        color='r'
       )

ax.plot(df_wgen1.DATE_TIME,
        df_wgen1.MODULE_TEMPERATURE.rolling(window=20).mean(),
        label='Module',
        color='c'
       )

ax.plot(df_wgen1.DATE_TIME,
        (df_wgen1.MODULE_TEMPERATURE-df_wgen1.AMBIENT_TEMPERATURE).rolling(window=20).mean(),
        label='Difference',
        color='m'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Tempreture and Module Tempreture over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('Tempreture')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen1['DATE_TIME'],
        df_wgen1['MODULE_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Module temperature',
       color='y')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date & Time vs. Module Tempreture')
plt.xlabel('Date & Time')
plt.ylabel('Module Tempreture')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen1['DATE_TIME'],
        df_wgen1['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=15,
        label='Ambient temperature',
       color='c')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date & Time vs. Ambient Tempreture')
plt.xlabel('Date & Time')
plt.ylabel('Ambient Tempreture')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen1['DATE_TIME'],
        df_wgen1['IRRADIATION'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=17,
        label='IRRADATION',
       color='r')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date & Time vs. Irradation')
plt.xlabel('Date & Time')
plt.ylabel('Irradation')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1['DATE_TIME'],
        df_pgen1['SOURCE_KEY'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='SOURCE_KEY',
       color='b')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date & Time vs. Inverter')
plt.xlabel('Date & Time')
plt.ylabel('Inverter')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1['DATE_TIME'],
        df_pgen1['AC_POWER'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC Power',
        color='m'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date & Time vs. AC Power')
plt.xlabel('Date & Time')
plt.ylabel('AC Power')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1['HOUR'],
        df_pgen1['AC_POWER'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC Power',
        color='m'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. AC Power')
plt.xlabel('Time')
plt.ylabel('AC Power')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1['DATE_TIME'],
        df_pgen1['DC_POWER'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=8,
        label='DC Power')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date & Time vs. DC Power')
plt.xlabel('Date & Time')
plt.ylabel('DC Power')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1['HOUR'],
        df_pgen1['DC_POWER'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=8,
        label='DC Power')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. DC Power')
plt.xlabel('Time')
plt.ylabel('DC Power')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1['DATE'],
        df_pgen1['AC_POWER'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='AC Power',
       color='g')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date vs. AC Power')
plt.xlabel('Date')
plt.ylabel('AC Power')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1['DATE'],
        df_pgen1['DC_POWER'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=8,
        label='DC Power',
       color='g')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date vs. DC Power')
plt.xlabel('Date')
plt.ylabel('DC Power')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen1['DATE'],
        df_wgen1['IRRADIATION'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='IRRADATION',
       color='k')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date vs. Irradation')
plt.xlabel('Date')
plt.ylabel('Irradation')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen1['HOUR'],
        df_wgen1['IRRADIATION'],
        marker='o',
        linestyle='',
        alpha=.3,
        ms=10,
        label='IRRADATION',
       color='c')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Irradation')
plt.xlabel('Time')
plt.ylabel('Irradation')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen1.AMBIENT_TEMPERATURE,
        df_wgen1.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.4,
        ms=10,
        label='Module Temperature (centigrade)',
       color='c')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Tempreture vs. Module Tempreture')
plt.xlabel('Ambient Tempreture')
plt.ylabel('Module Tempreture')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_pgen1.AC_POWER,
        df_pgen1.DC_POWER,
        marker='o',
        linestyle='',
        alpha=.4,
        ms=10,
        label='DC_POWER',
       color='g')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('AC POWER vs. DC POWER')
plt.xlabel('AC Power')
plt.ylabel('DC Power')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen1['IRRADIATION'],
        df_wgen1['MODULE_TEMPERATURE']+df_wgen1['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5, #transparency
        ms=10, #size of the dot
        label='temperature (Module + Ambient)',
       color='r')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradiation vs. Tempreture ')
plt.xlabel('Irradiation')
plt.ylabel('Tempreture')
plt.show()


result_left = pd.merge(df_pgen1,df_wgen1, on='DATE_TIME',how='left') #left, right, outer, inner
result_right = pd.merge(df_pgen1,df_wgen1, on='DATE_TIME',how='right') #left, right, outer, inner
result_inner = pd.merge(df_pgen1,df_wgen1, on='DATE_TIME',how='inner') #left, right, outer, inner
result_outer = pd.merge(df_pgen1,df_wgen1, on='DATE_TIME',how='outer') 
result_outer = pd.merge(df_pgen1,df_wgen1, on='DATE_TIME',how='outer') 
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(result_outer.IRRADIATION,
        result_outer.DC_POWER + result_outer.AC_POWER,
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='POWER',
       color='c')

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Power vs. Irradiation')
plt.xlabel('Irradiation')
plt.ylabel('Power')
plt.show()
df_wgen2['DATE_TIME'] = pd.to_datetime(df_wgen2['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df_wgen2['DATE'] = df_wgen2['DATE_TIME'].apply(lambda x:x.date())
df_wgen2['TIME'] = df_wgen2['DATE_TIME'].apply(lambda x:x.time())
df_wgen2['DATE'] = pd.to_datetime(df_wgen2['DATE'],format = '%Y-%m-%d').dt.date
df_wgen2['HOUR'] = pd.to_datetime(df_wgen2['TIME'],format='%H:%M:%S').dt.hour
df_wgen2['MINUTES'] = pd.to_datetime(df_wgen2['TIME'],format='%H:%M:%S').dt.minute

df_wgen2
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['DATE_TIME'],
        df_wgen2['MODULE_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Module temperature',
       color='r')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date & Time vs. Module Tempreture')
plt.xlabel('Date & Time')
plt.ylabel('Module Tempreture')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['DATE_TIME'],
        df_wgen2['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Ambient temperature',
       color='r')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date & Time vs. Ambient Tempreture')
plt.xlabel('Date & Time')
plt.ylabel('Ambient Tempreture')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['DATE_TIME'],
        df_wgen2['IRRADIATION'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Irradiation',
       color='r')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date & Time vs. Irradiation')
plt.xlabel('Date & Time')
plt.ylabel('Irradiation')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['DATE'],
        df_wgen2['MODULE_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Module temperature',
       color='g')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date vs. Module Tempreture')
plt.xlabel('Date')
plt.ylabel('Module Tempreture')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['DATE'],
        df_wgen2['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Ambient temperature',
       color='g')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date vs. Ambient Tempreture')
plt.xlabel('Date')
plt.ylabel('Ambient Tempreture')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['DATE'],
        df_wgen2['IRRADIATION'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Irradiation',
       color='g')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Date vs. Irradiation')
plt.xlabel('Date')
plt.ylabel('Irradiation')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['HOUR'],
        df_wgen2['MODULE_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Module Temperature',
        color='m'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Module Temperature')
plt.xlabel('Time')
plt.ylabel('Module Temperature')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['HOUR'],
        df_wgen2['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Ambient Temperature',
        color='m'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Ambient Temperature')
plt.xlabel('Time')
plt.ylabel('Ambient Temperature')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['HOUR'],
        df_wgen2['IRRADIATION'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Irradiation',
        color='m'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Irradiation')
plt.xlabel('Time')
plt.ylabel('Irradiation')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['MINUTES'],
        df_wgen2['MODULE_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Module Temperature',
        color='y'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Module Temperature')
plt.xlabel('Time')
plt.ylabel('Module Temperature')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['MINUTES'],
        df_wgen2['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Ambient Temperature',
        color='y'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Ambient Temperature')
plt.xlabel('Time')
plt.ylabel('Ambient Temperature')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['MINUTES'],
        df_wgen2['IRRADIATION'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Irradiation',
        color='y'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Irradiation')
plt.xlabel('Time')
plt.ylabel('Irradiation')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['HOUR']+df_wgen2['MINUTES'],
        df_wgen2['MODULE_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Module Temperature',
        color='c'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Module Temperature')
plt.xlabel('Time')
plt.ylabel('Module Temperature')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['HOUR']+df_wgen2['MINUTES'],
        df_wgen2['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Ambient Temperature',
        color='c'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Ambient Temperature')
plt.xlabel('Time')
plt.ylabel('Ambient Temperature')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['HOUR']+df_wgen2['MINUTES'],
        df_wgen2['IRRADIATION'],
        marker='o',
        linestyle='',
        alpha=.5,
        ms=10,
        label='Irradiation',
        color='c'
       )
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Time vs. Irradiation')
plt.xlabel('Time')
plt.ylabel('Irradiation')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2.AMBIENT_TEMPERATURE,
        df_wgen2.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.4,
        ms=10,
        label='Module Temperature (centigrade)',
       color='b')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Tempreture vs. Module Tempreture')
plt.xlabel('Ambient Tempreture')
plt.ylabel('Module Tempreture')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2.AMBIENT_TEMPERATURE,
        df_wgen2.IRRADIATION,
        marker='o',
        linestyle='',
        alpha=.4,
        ms=10,
        label='Irradiation',
       color='b')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Tempreture vs. Irradiation')
plt.xlabel('Ambient Tempreture')
plt.ylabel('Irradition')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2.MODULE_TEMPERATURE,
        df_wgen2.IRRADIATION,
        marker='o',
        linestyle='',
        alpha=.4,
        ms=10,
        label='Irradiation',
       color='b')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Tempreture vs. Irradiation')
plt.xlabel('Module Tempreture')
plt.ylabel('Irradition')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2['IRRADIATION'],
        df_wgen2['MODULE_TEMPERATURE']+df_wgen2['AMBIENT_TEMPERATURE'],
        marker='o',
        linestyle='',
        alpha=.5, #transparency
        ms=10, #size of the dot
        label='temperature (Module + Ambient)',
       color='r')
ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradiation vs. Tempreture ')
plt.xlabel('Irradiation')
plt.ylabel('Tempreture')
plt.show()
dates = df_wgen2['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_wgen2[df_wgen2['DATE']==date]

    ax.plot(df_data.AMBIENT_TEMPERATURE,
            df_data.MODULE_TEMPERATURE,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=12,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Tempreture vs. Ambient Tempreture')
plt.xlabel('Ambient Tempreture')
plt.ylabel('Module Tempreture')
plt.show()

dates = df_wgen2['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    df_data = df_wgen2[df_wgen2['DATE']==date]

    ax.plot(df_data.IRRADIATION,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Irradition Per Day')
plt.xlabel('Irradition')
plt.ylabel('')
plt.show()
_, ax = plt.subplots(1, 1, figsize=(18, 9))

ax.plot(df_wgen2.DATE_TIME,
        df_wgen2.AMBIENT_TEMPERATURE.rolling(window=20).mean(),
        label='Ambient',
        color='r'
       )

ax.plot(df_wgen2.DATE_TIME,
        df_wgen2.MODULE_TEMPERATURE.rolling(window=20).mean(),
        label='Module',
        color='c'
       )

ax.plot(df_wgen2.DATE_TIME,
        (df_wgen2.MODULE_TEMPERATURE-df_wgen2.AMBIENT_TEMPERATURE).rolling(window=20).mean(),
        label='Difference',
        color='m'
       )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Ambient Tempreture and Module Tempreture over 34 Days')
plt.xlabel('Date and Time')
plt.ylabel('Tempreture')
plt.show()