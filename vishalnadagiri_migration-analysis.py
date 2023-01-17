# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_can = pd.read_excel('../input/migration-data-canada/Canada.xlsx',sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skip_footer=2)
df_can.head()
df_can.shape
df_can.columns.values
#Drop unnecessary columns 
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)
# df_can.head(2)
print(df_can.shape)
df_can.head(2)
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
df_can.columns
df_can['Total'] = df_can.sum(axis=1)
#Lets check for missing values 
df_can.isnull().sum()
#So no nul values. Now lets see quick summary of each feature 
df_can.describe()
df_can.set_index('Country', inplace=True)
df_can.head(3)
df_can.index.name = None
df_can.head()
df_can.columns = list(map(str, df_can.columns))
# useful for plotting later on
years = list(map(str, range(1980, 2014)))
years
import matplotlib.pyplot as plt
%matplotlib inline 
ind_chi =df_can.loc[['India','China'], years].transpose()
ind_chi
ind_chi.index=ind_chi.index.map(int)
ind_chi.plot(kind='Line')
plt.title('India and China Comparision')
plt.xlabel('Year')
plt.ylabel('Number of Immigrants')
df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)
df_top5 = df_can.head(5)
df_top5 = df_top5[years].transpose() 
print(df_top5)
df_top5.index = df_top5.index.map(int)
df_top5.plot(kind='line', figsize=(14, 8))
plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')
ax = df_top5.plot(kind='area', alpha=0.35, figsize=(20, 10))

ax.set_title('Immigration Trend of Top 5 Countries')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')
count, bin_edges = np.histogram(df_top5, 15)

# un-stacked histogram
df_top5.plot(kind ='hist', 
          figsize=(10, 6),
          bins=15,
          alpha=0.6,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen']
         )

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()
df_can.sort_values(by='Total', ascending=False, axis=0, inplace=True)
df_t3 = df_can.head(3)
df_t3 = df_t3[years].transpose() 
count, bin_edges = np.histogram(df_t3, 15)

# un-stacked histogram
df_t3.plot(kind ='hist', 
          figsize=(10, 6),
          bins=15,
          alpha=0.6,
          xticks=bin_edges,
          color=['coral', 'darkslateblue', 'mediumseagreen'],
          stacked=True,#can be made false to see the Transparency of frequency distribution by a perticular country
#           xlim=(xmin, xmax)
         )

plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()
df_India = df_can.loc['India', years]
df_India.head()
df_India.plot(kind='bar', figsize=(10, 6))

plt.xlabel('Year') # add to x-label to the plot
plt.ylabel('Number of immigrants') # add y-label to the plot
plt.title('Indians immigrants to Canada from 1980 to 2013') # add title to the plot

plt.show()
df_India.plot(kind='bar', figsize=(10, 6))

plt.xlabel('Year') # add to x-label to the plot
plt.ylabel('Number of immigrants') # add y-label to the plot
plt.title('Indians immigrants to Canada from 1980 to 2013') # add title to the plot

# Annotate arrow
plt.annotate('',                      # s: str. will leave it blank for no text
             xy=(13, 23500),             # place head of the arrow at point (year 2012 , pop 70)
             xytext=(5, 9000),         # place base of the arrow at point (year 2008 , pop 20)
             xycoords='data',         # will use the coordinate system of the object being annotated 
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
            )

#Annotate Text
plt.annotate('Rise in Immigration', # text to display
             xy=(6, 12500),                    # start the text at at point (year 2008 , pop 30)
             rotation=44,                  # based on trial and error to match the arrow
             va='bottom',                    # want the text to be vertically 'bottom' aligned
             ha='left',                      # want the text to be horizontally 'left' algned.
            )
