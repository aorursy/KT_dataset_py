import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import os
%matplotlib inline
print(os.listdir('../input'))
df=pd.read_csv('../input/current-employment-statistics-beginning-1990.csv')
df.head(3)
df = df.iloc[::-1]
df.head()
df.info()
df['Year'].unique()
df['Area Name'].unique()
df['Title'].unique()
df['Time'] = df['Month'].map(str) +'-'+ df['Year'].map(str)
df.head()
def total_nonfarm_in_an_area(area, title, df_nonfarm):

    df_nonfarm = df[df['Title']==title]

    temp_df=df_nonfarm[df_nonfarm['Area Name']==area][['Current Employment','Time']]

    temp_df = temp_df.set_index('Time')

    temp_df.plot(figsize=(15,8))
total_nonfarm_in_an_area('New York City','Total Nonfarm',df)
total_nonfarm_in_an_area('Statewide','Total Nonfarm',df)