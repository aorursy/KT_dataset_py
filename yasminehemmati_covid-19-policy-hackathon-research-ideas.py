# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/covid19-in-usa/us_covid19_daily.csv')
data.head(12)
data.shape
yesterday = datetime.now() 

# datetime.now() starts with monday at 0 but for my analysis I will start at 1, in addition data given is one day behind present day 

yesterday.weekday() 
datacopyy= data.copy()

weekday = yesterday.weekday()  # add one since datime module starts with monday at zero and since data is being updated constantly

j= weekday #since first entry for date is a Tuesday (June 15 2020)

for i in range(datacopyy.shape[0]):   

    j= j%7 

    datacopyy.loc[[i], ['date']]=j

    j=j-1 # since data in decreasing order



datacopy = datacopyy.head(65)  #so data from the last 65 days since the dataset contains info from the last 143 days

datacopy.head(12)

    
#find the average number of people on ventilators per weekday:

# average number of people hospitalized increase per weekday ''

# average number of people that die more on days 'deathIncrease'



df_by_date = datacopy.groupby(['date']).mean()

df_by_date
df_by_date.columns
import matplotlib.pyplot as plt

### AVERAGE NUMBER OF PEOPLE IN ICU PER WEEKDAY

#plt.plot([0,1,2,3,4,5,6], df_by_date['onVentilatorCurrently'])

date = [0,1,2,3,4,5,6]

df1 = pd.DataFrame({'date': date,

                   'inIcuCurrently': df_by_date["inIcuCurrently"]})



plt.plot(date, df_by_date["inIcuCurrently"], 'o', color='black');df1.plot.bar(rot=0)
##Number of People currently hospitalized per weekday

df2 = pd.DataFrame({'date': date,

                   'hospitalizedCurrently': df_by_date["hospitalizedCurrently"]})

plt.plot(date, df_by_date['hospitalizedCurrently'], 'o', color='black');df2.plot.bar(rot=0)
##Number of People died per weekday

df3 = pd.DataFrame({'date': date,

                   'deathIncrease': df_by_date["deathIncrease"]})

plt.plot(date, df_by_date['deathIncrease'], 'o', color='black');df3.plot.bar(rot=0)
## average number of people on ventilator per weekday

df4 = pd.DataFrame({'date': date,

                   'onVentilatorCurrently': df_by_date["onVentilatorCurrently"]})

plt.plot(date, df_by_date['onVentilatorCurrently'], 'o', color='black');df4.plot.bar(rot=0)
## Increase in number of hospitalizations per weekday

df5 = pd.DataFrame({'date': date,

                   'hospitalizedIncrease': df_by_date['hospitalizedIncrease']})

plt.plot(date, df_by_date['hospitalizedIncrease'], 'o', color='black');df5.plot.bar(rot=0)