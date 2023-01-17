import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from pylab import figure, show, legend, ylabel

import matplotlib.lines as mlines

import matplotlib.pyplot as plt

import matplotlib.dates as mdates



import seaborn as sns



import numpy as np

from scipy import stats



import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,  iplot

init_notebook_mode(connected=True)
bakery = pd.read_csv('../input/transactions-from-a-bakery/BreadBasket_DMS.csv')

bakery.head()
bakery.shape
#prints count of each item

print(bakery['Item'].value_counts().tail(10))
#prints count of each item

print(bakery['Item'].value_counts().head(10))
bakery['Item'].value_counts().head(6)
bakery[bakery['Transaction']==348]
bakery[bakery['Item']== 'NONE'].head(3)
bakery = bakery[bakery['Item'] != 'NONE']
bakery[bakery['Item'] == 'NONE']
bakery[bakery['Item'] == 'Adjustment']
bakery[bakery['Transaction'] == 938]
bakery = bakery[bakery['Item'] != 'Adjustment']
#we do not have mising values

bakery.isnull().sum()
bakery['Date'].min()
bakery['Date'].max()
print(bakery.head())

bakery.dtypes

#Date is an object, need to chage to proper date
#covert to datetime

bakery['Date_Time'] = pd.to_datetime(bakery['Date'].apply(str)+' '+bakery['Time'],format="%Y/%m/%d %H:%M:%S")



#todatetime 

print(bakery.dtypes)
bakery.head()
bakery['Day_of_Week'] = bakery['Date_Time'].dt.weekday_name
bakery.to_excel('cleanBakeryDF.xlsx', sheet_name='Sheet1')
bakery['Month'] = bakery['Date_Time'].dt.month
#First month should be October

bakery['Month'].head(1)
#Last month should be April

bakery['Month'].tail(1)
#Dictionary to map months in order

mo = {10 : 1, 11 : 2, 12 : 3 , 1 : 4 , 2 : 5 , 3 : 6 , 4 : 7}

 

m = bakery['Month']

bakery['Month_Order'] = m.map(mo)
#adding season

##Dictionary to map month to season

x = {1 : 'Winter', 2 :'Winter', 3 :'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Autumn',10:'Autumn',11:'Autumn',12:'Winter'}



y = bakery['Month']

bakery['Season'] = y.map(x)
bakery.head(2)
bakery.loc[(pd.to_datetime(bakery['Date_Time'].dt.date) == '2016-10-30')].groupby(['Item'])['Transaction'].count().plot.bar()
bakery[bakery['Time'] == '01:21:05']
bakery['Hour'] = bakery['Date_Time'].dt.hour
bakery['Hour'].value_counts()
#Dictionary to map session

t = {7 : 'Morning', 8 :'Morning', 9 :'Morning',10:'Morning',11:'Morning',12:'Morning',13:'Afternoon',14:'Afternoon',15:'Afternoon',16:'Afternoon',17:'Afternoon',18:'Afternoon',19:'Evening',20:'Evening',21:'Evening',22:'Evening',23:'Evening'}



h = bakery['Hour']

bakery['Session'] = h.map(t)
#adding categories to items

from __future__ import print_function

from os.path import join, dirname, abspath

import xlrd



d = {}

wb = xlrd.open_workbook('../input/item-dic/items_dictionary.xlsx')

sh = wb.sheet_by_name('sheet1') 



for i in range(sh.nrows):

    cell_value_id = sh.cell_value(i,0)

    cell_value_class = sh.cell_value(i,1)

    d[cell_value_id] = cell_value_class

    
it = bakery['Item']

bakery['Category'] = it.map(d)
bakery.head(1)
bakery.loc[(pd.to_datetime(bakery['Date_Time'].dt.date) == '2016-10-30')].groupby(['Category'])['Transaction'].count().plot.bar()
bakery['Hourly'] = bakery['Date_Time'].dt.to_period('H')
bakery['Hourly'] = pd.to_datetime(bakery['Hourly'].apply(str),format="%Y/%m/%d %H:%M:%S")
bakery['Monthly'] = pd.to_datetime(bakery['Date_Time']).dt.to_period('M')
bakery['Weekly'] = pd.to_datetime(bakery['Date_Time']).dt.to_period('W')
temp_data = pd.read_csv('../input/temperature-data/temp_data.csv')
temp_data['Hourly'] = pd.to_datetime(temp_data['Hourly'].apply(str), format = '%Y/%m/%d %H:%M:%S')
temp_data.head()
temp_data.isnull().sum()
dates_for_graph =bakery['Date'].value_counts().sort_index()
bakery_temp = pd.merge(bakery, temp_data, on='Hourly', how='left')
bakery_temp.describe()
bakery_temp.fillna(method='ffill', inplace=True)
bakery_temp.groupby('Date')['temperature'].min().head(10)
trace1 = go.Scatter(

    x = bakery_temp.groupby('Date')['Item'].count().index,

    y = bakery_temp.groupby('Date')['Item'].count().values,

    mode = 'lines+markers',

    name = 'lines+markers')



data = [trace1]

layout = go.Layout(title = 'Daily Sales')

fig = go.Figure(data = data, layout = layout)

iplot(fig)
fig1 = figure(figsize=(15,6))

 

# and the first axes using subplot populated with data 

ax1 = fig1.add_subplot(1,1,1)

ax1.set_title("Sales and Temperature")

line1 = ax1.plot(bakery_temp.groupby('Date')['Item'].count(), 'o-')



ylabel("Total Sales per day")

 

# now, the second axes that shares the x-axis with the ax1

ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)

line2 = ax2.plot(bakery_temp.groupby('Date')['temperature'].min(), 'xr-')

ax2.yaxis.tick_right()

ax2.yaxis.set_label_position("right")

ylabel("Mininum temperature per day")





blue_line = mlines.Line2D([], [], color='blue', marker='o',

                          markersize=6, label='Sales')



red_line = mlines.Line2D([], [], color='red', marker='*',

                          markersize=6, label='Temperature')

plt.legend(handles=[blue_line,red_line])

#no working but leave it as it removes the axix labels

ax1.get_xaxis().set_major_locator(mdates.MonthLocator(interval=1))



show()


fig1 = figure(figsize=(15,6))

 

#the first axes using subplot populated with data 

ax1 = fig1.add_subplot(111)

ax1.set_title("Hot drinks sales and Temperature")

line1 = ax1.plot(bakery_temp[bakery_temp['Item'].isin(['Coffee', 'Tea', 'Hot Chocolate'])] .groupby('Date')['Item'].count(),'o-')

ylabel("Total Sales per day")

 

# now, the second axes that shares the x-axis with the ax1

ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)

line2 = ax2.plot(bakery_temp.groupby('Date')['temperature'].min(), 'xr-')

ax2.yaxis.tick_right()

ax2.yaxis.set_label_position("right")

ylabel("Mininum temperature per day")



#inverted Axis

ax2.invert_yaxis()



blue_line = mlines.Line2D([], [], color='blue', marker='o',

                          markersize=6, label='Sales')



red_line = mlines.Line2D([], [], color='red', marker='*',

                          markersize=6, label='Temperature')

plt.legend(handles=[blue_line,red_line])



ax1.get_xaxis().set_major_locator(mdates.MonthLocator(interval=1))



show()
item_hour = bakery_temp.groupby('Hourly')['Item'].count().values



temp_hour = bakery_temp.groupby('Hourly')['temperature'].min().values
hot_drink_df = bakery_temp[bakery_temp['Item'].isin(['Coffee', 'Tea', 'Hot Chocolate'])]
hot_drink_item_hour = hot_drink_df.groupby('Hourly')['Item'].count().values

hot_drink_temp_hour = hot_drink_df.groupby('Hourly')['temperature'].min().values
correlation, p_value = stats.pearsonr(temp_hour,item_hour)

print(correlation)

print(p_value)
plt.scatter(temp_hour, item_hour)
correlation, p_value = stats.pearsonr(hot_drink_temp_hour,hot_drink_item_hour)

print(correlation)

print(p_value)
plt.scatter(hot_drink_temp_hour, hot_drink_item_hour)
bakery_temp.groupby(['Date','Day_of_Week'])['Item'].count().sort_values()
bakery_temp[bakery_temp['Date'] == '2017-01-01']
bakery_temp.groupby('Day_of_Week')['Item'].count().plot.pie()
bakery_temp.groupby('Day_of_Week')['Item'].count().sort_values()
fig, axes = plt.subplots(3, 2, figsize=(15,6), sharex=True, sharey=True, squeeze=False )



fig.suptitle('Total Sales by Hour and Day', fontsize=12)

fig.text(0.06, 0.5, 'Total Item Sold', ha='center', va='center', rotation='vertical')

#fig.text(0.5, 0.04, 'Hours', ha='center', va='center')

Saturday = bakery_temp[bakery_temp['Day_of_Week'] == 'Saturday'].groupby('Hour')['Item'].count()

Saturday.plot(ax=axes[0][0], grid=True, kind='area', title='Saturday', xticks=range(6,24,1), yticks=range(0, 1000,200))



#Removing the item sold at 1:20 in the morning

Sunday = bakery_temp[(bakery_temp['Date'] != '2017-01-01') & (bakery_temp['Day_of_Week'] == 'Sunday')].groupby('Hour')['Item'].count()

Sunday.plot(ax=axes[0][1], grid=True, kind='area', title='Sunday', xticks=range(6,24,1), yticks=range(0, 1000,200))



Monday = bakery_temp[bakery_temp['Day_of_Week'] == 'Monday'].groupby('Hour')['Item'].count()

Monday.plot(ax=axes[1][0], grid=True, kind='area', title='Monday', xticks=range(6,24,1), yticks=range(0, 1000,200))



Tuesday = bakery_temp[bakery_temp['Day_of_Week'] == 'Tuesday'].groupby('Hour')['Item'].count()

Tuesday.plot(ax=axes[1][1], grid=True, kind='area', title='Tuesday', xticks=range(6,24,1), yticks=range(0, 1000,200))



Thursday = bakery_temp[bakery_temp['Day_of_Week'] == 'Thursday'].groupby('Hour')['Item'].count()

Thursday.plot(ax=axes[2][0], grid=True, kind='area', title='Thursday', xticks=range(6,24,1), yticks=range(0, 1000,200))



Friday = bakery_temp[bakery_temp['Day_of_Week'] == 'Friday'].groupby('Hour')['Item'].count()

Friday.plot(ax=axes[2][1], grid=True, kind='area', title='Friday', xticks=range(6,24,1), yticks=range(0, 1000,200))
bakery_temp.groupby('Date')['Item'].count().plot.box()
bakery_temp[(bakery_temp['Time'] > '20:00:00') & (bakery_temp['Day_of_Week'] == 'Saturday')].groupby(['Date','Day_of_Week','Time'])['Item'].count()
bakery_temp[(bakery_temp['Time'] > '18:00:00') & (bakery_temp['Day_of_Week'] == 'Friday')].groupby(['Date','Day_of_Week'])['Item'].count()
bakery_temp[(bakery_temp['Time'] > '17:00:00') & (bakery_temp['Day_of_Week'] == 'Friday')].groupby(['Date','Day_of_Week'])['Item'].count()
bakery_temp.shape
#unstack, will put the categories in columns

bakery_temp.groupby(['Month_Order','Category'])['Category'].count().unstack()


fig, ax = plt.subplots()

bakery_temp.groupby(['Month_Order','Category'])['Category'].count().unstack().plot(kind='bar', figsize=(15,6), ax=ax)

ax.set_title('Monthly Sales', fontsize=21, y=1.01)

ax.legend(loc="upper right")

ax.set_ylabel('Sales', fontsize=16)

ax.set_xlabel('Category', fontsize=16)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.show()
bakery_temp.groupby('Month')['Date'].nunique()
#this will give me a list with unique names of item

item_Name = bakery_temp['Item'].value_counts().index

#this will give me the values of the unique item name

item_Value = bakery_temp['Item'].value_counts().values
#this will give me a list with unique names of item

bakery_saturday = bakery_temp[bakery_temp['Day_of_Week'] == 'Saturday']

item_Name_Saturday = bakery_saturday['Item'].value_counts().index

#this will give me the values of the unique item name

item_Value_Saturday = bakery_saturday['Item'].value_counts().values
#this will give me a list with unique names of item

bakery_monday = bakery_temp[bakery_temp['Day_of_Week'] == 'Monday']

item_Name_monday = bakery_monday['Item'].value_counts().index

#this will give me the values of the unique item name

item_Value_monday = bakery_monday['Item'].value_counts().values
item_Value_Saturday[10:].sum()
item_Value_monday[:10]
item_Name_monday[:10]
item_Value_Saturday[:10]
item_Name_Saturday[:10]
#Top 10 items plus aggregating the rest as others
item_Saturday_Value = [1103,  760,  288,  246,  166,  161,  146,  146,  143,  118, 1328]
item_Saturday_Name = ['Coffee', 'Bread', 'Tea', 'Cake', 'Pastry', 'Sandwich', 'Hot chocolate',

       'Scone', 'Medialuna', 'Scandinavian', 'Other']
plt.figure(figsize=(12,4))

plt.ylabel('Values', fontsize='medium')

plt.xlabel('Items', fontsize='medium')

plt.title('10 Most sold itme')

plt.bar(item_Name[:10],item_Value[:10], width = 0.7, color="blue",linewidth=0.4)



plt.xticks(rotation=45)

plt.show()
init_notebook_mode(connected=True)



labels = item_Name_Saturday[:10]

values = item_Value_Saturday[:10]



trace = go.Pie(labels=labels, values=values)



data= [trace]

layout = go.Layout(title = 'Top 10 item sold on Saturday')

fig = go.Figure(data = data, layout = layout)

iplot(fig)


labels = item_Saturday_Name

values = item_Saturday_Value



trace = go.Pie(labels=labels, values=values)



data= [trace]

layout = go.Layout(title = 'All Items sold on Saturday')

fig = go.Figure(data = data, layout = layout)

iplot(fig)
bakery_temp['Session'].value_counts()
bakery_temp.to_pickle('bakery_temp_dataframe.pkl')
#Extract dates as we want then to be the index

dates = pd.DatetimeIndex(bakery_temp['Date_Time'])
bakery_temp_sum = bakery_temp[['Date_Time','Date','Item','Day_of_Week','temperature']].copy()
bakery_temp_sum.head(10)
bakery_temp_sum.to_pickle('bakery_temp_sum_dataframe.pkl')