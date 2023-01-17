import numpy as np

import pandas as pd
data = pd.read_excel('../input/DIM_MATCH.xlsx')
data
data.head()
data.info()
data.isnull().sum()
data=data.dropna(axis=0,how='any')



data.isnull().sum()
data.shape
print(data.columns)



for i in data.columns:

    print(i)
data.Win_Type.count()
data.iloc[[2,4,7],[0,3,4]]
data
data
data.columns
data = pd.read_excel('../input/DIM_MATCH.xlsx')



data.set_index("City_Name",inplace=True)



data.head()



data.loc["Bangalore"]
data = pd.read_excel('../input/DIM_MATCH.xlsx')



print(type(data.values))

print(type(data.keys()))
data.set_index('City_Name',inplace=True)
data
type(data.loc['Bangalore'])
data.loc[['Bangalore','Delhi'],['Team1','Team2','match_winner']]
data.head()
data.loc[data['match_winner']=="Kolkata Knight Riders",['Win_Type']]
data.head()
data.loc[data['match_winner']=='Rising Pune Supergiants','Win_Type':'Win_Margin']
data.head()
data=pd.read_excel('../input/DIM_MATCH.xlsx')



data=data.dropna(axis=0,how='any')

data.isnull().sum()



data.loc[data['Toss_Winner'].str.endswith("Riders"),'match_winner']
data=pd.read_excel('../input/DIM_MATCH.xlsx')



data=data.dropna(axis=0,how='any')

data.isnull().sum()



data.head()



data.loc[data['City_Name'].isin(['Kolkata','Hyderabad','Rajkot']), 'Venue_Name']
data.head()



data.loc[data['Venue_Name'].str.endswith('Gardens') & (data['match_winner']=='Kolkata Knight Riders') & data['Season_Year']==2015].count()