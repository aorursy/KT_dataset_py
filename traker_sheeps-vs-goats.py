#   Processing
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
import re
#   Visuals
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize']=(20,10)
df = pd.read_csv("../input/commodity_trade_statistics_data.csv", na_values=["No Quantity",0.0,''],sep=',')
df.head()
df.count()
df.isnull().sum()
df = df.dropna(how='any').reset_index(drop=True)  
df.isnull().sum()
df['commodity'].unique()[:5]
dfSheeps = df[df['commodity']=='Sheep, live'].reset_index(drop=True)  
dfGoats = df[df['commodity']=='Goats, live'].reset_index(drop=True)  
dfSheeps.head()
dfSheepsGrouped = pd.DataFrame({'weight_kg' : dfSheeps.groupby( ["year","flow","commodity"] )["weight_kg"].sum()}).reset_index()
dfGoatsGrouped = pd.DataFrame({'weight_kg' : dfGoats.groupby( ["year","flow","commodity"] )["weight_kg"].sum()}).reset_index()
dfSheepsGrouped.head()
f, ax = plt.subplots(1, 1)
dfgr = pd.concat([dfSheepsGrouped,dfGoatsGrouped])
ax = sns.pointplot(ax=ax,x="year",y="weight_kg",data=dfgr[dfgr['flow']=='Import'],hue='commodity')
_ = ax.set_title('Global imports of kgs by animal')
dfSheeps.head()
dfSheepsGrouped = pd.DataFrame({'weight_kg' : dfSheeps.groupby( ["country_or_area","flow","commodity"] )["weight_kg"].sum()}).reset_index()
dfSheepsGrouped.head()
sheepsImportsCountry = dfSheepsGrouped[dfSheepsGrouped['flow']=='Import']
sheepsExportsCountry = dfSheepsGrouped[dfSheepsGrouped['flow']=='Export']
sheepsImportsCountry.head()
ax = sns.barplot(x="weight_kg", y="country_or_area", data=sheepsImportsCountry.sort_values('weight_kg',ascending=False)[:15])
_ = ax.set(xlabel='Kgs', ylabel='Country or area',title = "Countries or areas that imported more kgs of Sheeps")
ax = sns.barplot(x="weight_kg", y="country_or_area", data=sheepsExportsCountry.sort_values('weight_kg',ascending=False)[:15])
_ = ax.set(xlabel='Kgs', ylabel='Country or area',title = "Countries or areas that exported more kgs of Sheeps")