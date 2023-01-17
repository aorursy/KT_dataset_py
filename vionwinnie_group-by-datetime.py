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
import datetime
df = pd.DataFrame(    

     {'COMM_FILE_ID': ['623270','623270','623270'], 'FILE_ID': [17460152,20000,30000], 'START_DATE': [datetime.datetime(2020, 3, 3, 11, 19, 9, 3260),datetime.datetime(2020, 3, 5, 11, 19, 9, 3260),datetime.datetime(2020, 3, 6, 11, 19, 9, 3260)]})

df['date'] = pd.to_datetime(df.START_DATE).values.astype(np.int64)

df['year'] = df.START_DATE.dt.year

df['month'] = df.START_DATE.dt.month

df['day']=df.START_DATE.dt.day
df2 = pd.DataFrame(pd.to_datetime(df.groupby('COMM_FILE_ID').mean().date))

df2