# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')

data.info()
data.columns
data.shape 
data.head()
#Shows first five datas
data.tail()
#Shows last five datas
data.corr()
#It gives us corrolation between dataframes
#If number close to 1, there is right proportion between dataframes
#If number close to - 1, there is inverse ratio between dataframes
#If number is 0, dataframes are irrelevant.
#CORRELATION MAP

f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
#You can use "-.", ":", "-", "--" for linestyle
# "gca" stands for 'get current axis'

ax = plt.gca()
data.plot(kind='line',x='Year_of_Release',y='NA_Sales',linewidth=1,grid=True,linestyle = '-.', ax=ax)
data.plot(kind='line',x='Year_of_Release',y='JP_Sales', color='red',linewidth=1, grid=True,linestyle = ':', ax=ax)
plt.legend(loc='upper right')
plt.title("Comparison of North America Sales and Japan Sales")
plt.show()

#Scatter Plot
#Scatter plot is better when there is a corrolation between dataframes
# x = global sales, y = na sales
data.plot(kind='scatter', x='NA_Sales', y='Global_Sales',alpha = 0.5,color = 'red')
plt.xlabel('North America Sales')              
plt.ylabel('Global Sales')
plt.title('Sales of North America and Global Scatter Plot')            # title = title of plot
# Histogram
# bins = number of bar in figure
#color types= b:blue, c:cyan, r:red, g:green, y:yellow
data.Critic_Score.plot(kind = 'hist',color="r", bins = 50,figsize = (12,12))
plt.show()
fig, axes = plt.subplots(nrows=2,ncols=1)
data.plot(kind = "hist",y = "Critic_Score",bins = 50,range= (0,100),normed = True,ax = axes[0])
data.plot(kind = "hist",y = "Critic_Score",bins = 50,range= (0,100),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt
data.plot()
#so confusing
data.plot(subplots = True)
plt.show()
#Filtering
data[np.logical_and(data['NA_Sales']>10, data['EU_Sales']>10 )]
#If we want, we can use "&&" instead of comma
data["Developer_on_what"] = data.Developer + " "+ data.Platform
data
meanTotalSales = sum(data.Global_Sales)/len(data.Global_Sales)
data["Global_Sales_Status"] = ["above average" if i > meanTotalSales else "average below" for i in data.Global_Sales]
data.loc[:, ['Name', 'Global_Sales', 'Global_Sales_Status']]
data.describe()
# count = Number of entries
# mean = Avarage of entries
# std = Standart deviation
# min = Mininmum entry
# 25% = First quartile
# 50% = Second quartile or median
# 75% = Third quartile
# max = Maximum entry
print(data['Publisher'].value_counts(dropna =False))

data.boxplot(column='Global_Sales',by = 'Global_Sales_Status', figsize=(10,10))
df_new=data.head() #make new dataframe
df_new
# melt func
# id_vars is the variable which need to be left unaltered which is “Name”
# value_vars is what we want to melt
melted_df = pd.melt(frame=df_new,id_vars = 'Name', value_vars= ['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales'])
melted_df
melted_df.pivot(index = 'Name', columns = 'variable',values='value')
#returns previous shape
conct1=data.head()
conct2=data.tail()
#If we want concatenate two dataframes:

conc_dataframe = pd.concat([conct1,conct2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
# ignore_index: boolean, default False. If True, do not use the index values on the concatenation axis
#If False, use previous index values
conc_dataframe
data.dtypes
data['Name'] = data['Name'].astype('category')
data["Year_of_Release"].value_counts(dropna =False)
# As you see, there are 269 NaN values.
new_data=data
new_data["Year_of_Release"].dropna(inplace = True)
new_data["Year_of_Release"].value_counts(dropna =False)
# As you see there are no NaN values.
#data["Year_of_Release"] = data["Year_of_Release"].fillna(data["Year_of_Release"].mean())
#data["Year_of_Release"].value_counts(dropna =False)
#data['Year_of_Release'] = new_df['Year_of_Release'].astype('int')