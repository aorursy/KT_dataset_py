import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/cusersmarildownloadsfuneralscsv/funerals.csv', delimiter=';',encoding = "ISO-8859-1") 
df.dataframeName = 'funerals.csv'
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')
df.head()
df1=df.drop(["postcode_area_of_last_known_address", "cost_recovered", "date_referred_to_treasury_solicitor","gender"], axis = 1)
df1
df2=df1.rename(columns={"date_of_death": "ds", "cost_of_funeral": "y"})
df2
df2['y'] = df2['y'].str.replace('\£','')
df2
print(df2.dtypes)
df2.dropna()