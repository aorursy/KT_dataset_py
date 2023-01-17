import pandas as pd

calendar_data=pd.read_csv('../input/calendar.csv',sep=',')

calendar_data.head()
split=pd.DataFrame()

split['year']=pd.to_datetime(calendar_data['date']).dt.year

split['month']=pd.to_datetime(calendar_data['date']).dt.month

split['dayofweek']=pd.to_datetime(calendar_data['date']).dt.dayofweek

split.head()

new_calendar=calendar_data.join(split,on=None,how='left',sort=False)





#remove $ sign in price
new_calendar['price']=new_calendar['price'].str.replace('$','')
new_calendar['price']=pd.to_numeric(new_calendar['price'],errors='coerce')
import seaborn as sn

import matplotlib.pyplot as plt

%matplotlib inline

price_month=new_calendar[['month','price']]

price_month_mean=price_month.groupby('month').mean()

plot1=price_month_mean.plot(kind='bar')

plot1.set_xlabel('Month')

plot1.set_ylabel('Average price of listings ($)')

plot1
price_month1=price_month.pivot(columns='month',values='price')

plot2=price_month1.boxplot()

plot2.set_xlabel('Month')

plot2.set_ylabel('Price of listing ($)')

plot2.set_title('Listing price by month')
price_week=new_calendar[['dayofweek','price']]

price_week_mean=price_week.groupby('dayofweek').mean()

plot3=price_week_mean.plot(kind='bar',legend=None)

plot3.set_xlabel('Week (0 to 6:Monday to Sunday)')

plot3.set_ylabel('Average price of listings ($)')

plot3.set_title('Price of listings Vs day of week')

plot3
price_week_mean
price_month2=price_week.pivot(columns='dayofweek',values='price')

plot2=price_month2.boxplot()

plot2.set_xlabel('Week')

plot2.set_ylabel('Price of listing ($)')

plot2.set_title('Listing price by Week')