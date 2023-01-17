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
df = pd.read_csv('/kaggle/input/mortaliy-moscow-20102020/moscow_stats.csv')
df.info()
df.head(5)
df.describe()
df.plot.scatter(x ="Year", y = "StateRegistrationOfDeath")
df.plot.line(x ="ID", y = "StateRegistrationOfDeath") #what happend in 2010?
trans={"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,"July":7,"August":8,"September":9,"October":10, "November":11,"December":12}

def getdatetime(df):

    month = df['Month']

    year = df['Year']

    

    return str(year)+"."+str(trans.get(month))

    
df['time'] = df.apply(getdatetime, axis=1)
df['time'] = pd.to_datetime(df['time'])
df2 = df[['StateRegistrationOfDeath', 'time']]

from statsmodels.tsa.seasonal import seasonal_decompose
df2 = df2.set_index('time')
decomposed = seasonal_decompose(df2['StateRegistrationOfDeath'])
decomposed.trend.plot() # global trend
decomposed.seasonal.plot() #seasonal data