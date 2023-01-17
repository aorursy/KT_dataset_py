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
df=pd.read_csv('/kaggle/input/demodatapractice/Train.csv')
#Shows top five rows of dataframe

df.head()
# Info of data frame in terms of elements in each row

df.info()
#Calculate mean of each row

print(df.mean())
# calculate median of each row using

print(df.median())
#min and max value using

print(df['Age'].min())

print(df['Age'].max())
# FInd about the distribution of data

df.quantile(0.75)
#Print value count of each element in the column

df['Age'].value_counts()
# print sum of attrition rate for the employee of same age

df.groupby(df['Age'])['Attrition_rate'].sum()
#Setting limit to display rows and columns to none to see complete dataframe

#pd.options.display.max_rows=None

#pd.options.display.max_columns=None
df
print(df.sort_values('Age',ascending=False))
print(df.sort_values(['Age','Time_of_service'],ascending=[False,True]))
df.reset_index()
df=df.sort_index(level=['Age','Time_of_service'],ascending=[True,False])
df