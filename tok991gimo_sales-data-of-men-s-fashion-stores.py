#This is a dataset of Sales Data of Men's Fashion Stores in Netherlands
#References: Verbeek, Marno (2004) A Guide to Modern Econometrics, John Wiley and Sons, chapter 3
import numpy as np
import numpy.random
import pandas as pd
clothing=pd.read_csv('../input/Clothing.csv')
print(clothing)
print(clothing.describe())
print(clothing.head())
print(clothing.tail())
#group by hours worked
print(clothing.groupby(['hoursw']).sum())

#sort by gross-profit margin
print(clothing.sort_values(by='margin'))

#slice four rows 
print(clothing.iloc[:4])

#remove the last 4 rows

print(clothing.iloc[:-4])

#Correlation between investment in shop-premises and investment in automation
print(clothing['inv1'].corr(clothing['inv2']))
#Create a new columns

clothing['channel']=np.random.choice(['retail','online','direct'],400) #Sales channel

clothing['city']=np.random.choice(['Amsterdam','Rotterdam'],400) #Cities where the stores are

print(clothing)
#Pivot table highlighting the annual sales

print(clothing.pivot_table('tsale',index='channel',columns='city'))

#Set a multi-index with channel and city

clothing_0=clothing.copy()

print(clothing_0.set_index(['channel','city']))
#Area plot of the dataframe

import matplotlib.pyplot as plt 

clothing.plot.area(stacked=False)

plt.show()
#create new column with the order date where the sales channel is online

start=pd.datetime(2016,1,1)

clothing['orderdate']=pd.date_range(start,periods=400).where(clothing['channel']=='online')

print(clothing)
#Make a custom business calendar for the dates received

from pandas.tseries.offsets import CustomBusinessDay

weekmask_netherlands='Mon Tue Wed Thu Fri'

bdateneth=CustomBusinessDay(weekmask=weekmask_netherlands)

#Delivery takes between 5-10 days

delivery=start+pd.Timedelta(days=np.random.randint(5,10))

clothing['deliverydate']=pd.bdate_range(delivery,periods=400,freq=bdateneth).where(clothing['channel']=='online')

print(clothing)
#Check if the order dates and delivery dates are the same

print(clothing['orderdate']==clothing['deliverydate'])
#drop missing values 

print(clothing.dropna())
#plotting with seaborn

import seaborn as sns

sns.barplot(x='margin',y='inv2', data=clothing)

sns.despine()

plt.tight_layout()

plt.show()

#ols table  infereantial statsitcs 

import statsmodels.api as sm

mod=sm.OLS.from_formula('tsale~C(sales)+hoursw+inv2+ssize',data=clothing)

res=mod.fit()

print(res.summary())