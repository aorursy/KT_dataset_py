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
df1 = pd.read_csv("/kaggle/input/airline-2019/june2019/405557996_T_ONTIME_REPORTING.csv")
df2 = pd.read_csv("/kaggle/input/airline-2019/nov2019/405557996_T_ONTIME_REPORTING.csv")
df3 = pd.read_csv("/kaggle/input/airline-2019/feb2019/405557996_T_ONTIME_REPORTING.csv")
df4 = pd.read_csv("/kaggle/input/airline-2019/jul2019/405557996_T_ONTIME_REPORTING.csv")
df5 = pd.read_csv("/kaggle/input/airline-2019/oct2019/405557996_T_ONTIME_REPORTING.csv")
df6 = pd.read_csv("/kaggle/input/airline-2019/aug2019/405557996_T_ONTIME_REPORTING.csv")
df7 = pd.read_csv("/kaggle/input/airline-2019/Jan2019/405557996_T_ONTIME_REPORTING.csv")
df8 = pd.read_csv("/kaggle/input/airline-2019/dec20219/405557996_T_ONTIME_REPORTING.csv")
df9 = pd.read_csv("/kaggle/input/airline-2019/mar2019/405557996_T_ONTIME_REPORTING.csv")
df10 = pd.read_csv("/kaggle/input/airline-2019/apr2019/405557996_T_ONTIME_REPORTING.csv")
df11 = pd.read_csv("/kaggle/input/airline-2019/may2019/405557996_T_ONTIME_REPORTING.csv")
df12 = pd.read_csv("/kaggle/input/airline-2019/sept2019/405557996_T_ONTIME_REPORTING.csv")

df=pd.concat([df1, df2, df3, df4, df4, df5, df7, df8, df9, df10, df11, df12], ignore_index=True)
df.info()
df.ORIGIN_CITY_NAME.unique
df[['org_city_name', 'org_state_name']]=df.ORIGIN_CITY_NAME.str.split(",",expand=True)
df.info()
df1 = df.sort_values('CARRIER_DELAY',ascending = False).groupby('ORIGIN').head()
print (df1[['OP_CARRIER_AIRLINE_ID','ORIGIN', 'ORIGIN_CITY_NAME','CARRIER_DELAY']])
df1 = df.sort_values('CARRIER_DELAY',ascending = False).groupby('ORIGIN').head()
print (df1[['OP_CARRIER_AIRLINE_ID','ORIGIN', 'ORIGIN_CITY_NAME','CARRIER_DELAY']])