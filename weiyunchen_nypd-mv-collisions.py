import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import folium

import os

import warnings; warnings.filterwarnings(action='once')









#显示所有列

pd.set_option('display.max_columns', None)

#显示所有行

pd.set_option('display.max_rows', None)

#设置value的显示长度为50

pd.set_option('max_colwidth',50)



data=pd.read_csv('../input/nypd-motor-vehicle-collisions.csv')
data.head(3)
print(len(data))

data.describe()
def null_count(data):  # 定义 null 值查找函数，函数名 null_count

    null_data = data.isnull().sum()  # 查找各个特征 null 值并计算数量

    null_data = null_data.drop(null_data[null_data == 0].index).sort_values(

        ascending=False)  # 删除数目为零的特征，降序排列

    return null_data  # 返回结果



null_count(data)  # 调用 null_count 函数统计 data 的 null，输出结果
plt.figure(figsize=(10,7))

sns.heatmap(data.isnull(), cbar = False, cmap = 'viridis')
data[(data['NUMBER OF PERSONS INJURED'].isnull().values==True)&(data['NUMBER OF PERSONS KILLED'].isnull().values==True)&\

     (data['NUMBER OF PEDESTRIANS INJURED'].isnull().values==True)&(data['NUMBER OF PEDESTRIANS KILLED'].isnull().values==True)&\

     (data['NUMBER OF CYCLIST INJURED'].isnull().values==True)&(data['NUMBER OF CYCLIST KILLED'].isnull().values==True)&\

     (data['NUMBER OF MOTORIST INJURED'].isnull().values==True)&(data['NUMBER OF MOTORIST KILLED'].isnull().values==True)].head()#查看各类受伤或死亡人数为空值的数据信息
data[(data['CONTRIBUTING FACTOR VEHICLE 3'].isnull().values==True)&\

     ((data['CONTRIBUTING FACTOR VEHICLE 4'].notnull().values==True)|(data['CONTRIBUTING FACTOR VEHICLE 5'].notnull().values==True))].head()
data[(data['CONTRIBUTING FACTOR VEHICLE 3'].isnull().values==True)&\

     ((data['CONTRIBUTING FACTOR VEHICLE 4'].notnull().values==True)|(data['CONTRIBUTING FACTOR VEHICLE 5'].notnull().values==True))].head()
data['NUMBER OF PERSONS INJURED'].fillna(0,inplace=True) 

data['NUMBER OF PERSONS KILLED'].fillna(0,inplace=True) 

data['LATITUDE'].fillna('U',inplace=True) 

data['LONGITUDE'].fillna('U',inplace=True) 
data['NUMBER OF PERSONS INFLUENCED'] = data['NUMBER OF PERSONS INJURED'] + data['NUMBER OF PERSONS KILLED']
location = data['LOCATION'].value_counts()



count_loc = pd.DataFrame({"LOCATION" : location.index, "ValueCount":location})

count_loc.index = range(len(location))

count_loc.head()



loc = data.groupby('LOCATION').first()



new_loc = loc.loc[:, ['LATITUDE', 'LONGITUDE', 'ZIP CODE', 'ON STREET NAME', 'BOROUGH']]

new_loc.head()
the_loc = pd.merge(count_loc,new_loc,on='LOCATION')

the_loc.drop(the_loc.index[0], inplace=True)

the_loc.head()
nmap = folium.Map(location=[40.721757, -73.930529],

                        zoom_start=13)



for i in range(200):

    lat = the_loc.iloc[i][2]

    long = the_loc.iloc[i][3]

    radius = the_loc['ValueCount'].iloc[i] / 30

    

    if the_loc['ValueCount'].iloc[i] > 300:

        color = "#FF4500"

    else:

        color = "#008080"

    

    popup_text = """Lat : {}<br>

                Lng : {}<br>

                ZIP CODE : {}<br>

                ON STREET NAME : {}<br>

                BOROUGH : {}<br>

                Incidents : {}<br>"""

    popup_text = popup_text.format(the_loc['LATITUDE'].iloc[i],

                               the_loc['LONGITUDE'].iloc[i],

                               the_loc['ZIP CODE'].iloc[i],

                               the_loc['ON STREET NAME'].iloc[i],

                               the_loc['BOROUGH'].iloc[i],

                               the_loc['ValueCount'].iloc[i]

                               )

    folium.CircleMarker(location = [lat, long], popup= popup_text,radius = radius, color = color, fill = True).add_to(nmap)
nmap
def get_date(DATE):

    date = DATE[:-12]

    return date



data['date'] = data['DATE'].apply(get_date)
time_place_person = data.groupby(by=['BOROUGH','date'])['NUMBER OF PERSONS INFLUENCED'].sum().unstack().reset_index().melt(id_vars='BOROUGH')

time_place_person.head()
df1=time_place_person[time_place_person['BOROUGH']=='BRONX']

df1['value_BRONX']=df1['value']

df1.drop(columns=['value','BOROUGH'], inplace=True)



df2=time_place_person[time_place_person['BOROUGH']=='BROOKLYN']

df2['value_BROOKLYN']=df2['value']

df2.drop(columns=['value','BOROUGH'], inplace=True)



df3=time_place_person[time_place_person['BOROUGH']=='MANHATTAN']

df3['value_MANHATTAN']=df3['value']

df3.drop(columns=['value','BOROUGH'], inplace=True)



df4=time_place_person[time_place_person['BOROUGH']=='QUEENS']

df4['value_QUEENS']=df4['value']

df4.drop(columns=['value','BOROUGH'], inplace=True)



df5=time_place_person[time_place_person['BOROUGH']=='STATEN ISLAND']

df5['value_STATEN ISLAND']=df5['value']

df5.drop(columns=['value','BOROUGH'], inplace=True)





df_1 = pd.merge(df1,df2,on='date')

df_2 = pd.merge(df3,df4,on='date')

df_3 = pd.merge(df_1,df_2,on='date')

df = pd.merge(df5,df_3,on='date')
df.drop(80, inplace=True)
y_LL = 80

y_UL = int(df.iloc[:, 1:].max().max()*1.1)

y_interval = 80

mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:pink']    



# Draw Plot and Annotate

fig, ax = plt.subplots(1,1,figsize=(24, 6), dpi= 80)    



columns = df.columns[1:]  

for i, column in enumerate(columns):

    plt.plot(df.date.values, df[column].values, lw=1.5, color=mycolors[i])    

    plt.text(df.shape[0]+1, df[column].values[-9], column, fontsize=9, color=mycolors[i])



# Draw Tick lines  

for y in range(y_LL, y_UL, y_interval):    

    plt.hlines(y, xmin=0, xmax=80, colors='black', alpha=0.3, linestyles="--", lw=0.5)



# Decorations    

plt.tick_params(axis="both", which="both", bottom=False, top=False,    

                labelbottom=True, left=False, right=False, labelleft=True)        



# Lighten borders

plt.gca().spines["top"].set_alpha(.3)

plt.gca().spines["bottom"].set_alpha(.3)

plt.gca().spines["right"].set_alpha(.3)

plt.gca().spines["left"].set_alpha(.3)



plt.title('NY BOROUGH Traffic Accident', fontsize=12)

plt.yticks(range(y_LL, y_UL, y_interval), [str(y) for y in range(y_LL, y_UL, y_interval)], fontsize=9)    

plt.xticks(range(0, df.shape[0], 6), df.date.values[::6], horizontalalignment='left', fontsize=9)    

plt.ylim(y_LL, y_UL)    

plt.xlim(-1, 81)    

plt.show()
def get_month(DATE):

    month = DATE[5:-12]

    return month



# def get_date(DATE):

    # date = DATE[:-12]

    # return date



def get_hour(TIME):

    hour = TIME[0:-3]

    return hour
data['month'] = data['DATE'].apply(get_month)

# data['date'] = data['DATE'].apply(get_date)

data['hour'] = data['TIME'].apply(get_hour)
date_statistics = data['date'].value_counts()

hour_statistics = data['hour'].value_counts()

month_statistics = data['month'].value_counts()







date_1 = pd.DataFrame({"date" : date_statistics.index, "statistics":date_statistics})

date_1.index = range(len(date_statistics))



hour_1 = pd.DataFrame({"hour" : hour_statistics.index, "statistics":hour_statistics})

hour_1.index = range(len(hour_statistics))



month_1 = pd.DataFrame({"month" : month_statistics.index, "statistics":month_statistics})

month_1.index = range(len(month_statistics))







date_1['index']=date_1['date']

month_1['index']=month_1['month']

hour_1['index']=hour_1['hour']







date_0 = date_1.groupby('index').first()

hour_0 = hour_1.groupby('index').first()

month_0 = month_1.groupby('index').first()
date_0.drop('2019-03', inplace=True)



hour_0['hour'][0] = '00'

hour_0['hour'][1] = '01'

hour_0['hour'][12] = '02'

hour_0['hour'][17] = '03'

hour_0['hour'][18] = '04'

hour_0['hour'][19] = '05'

hour_0['hour'][20] = '06'

hour_0['hour'][21] = '07'

hour_0['hour'][22] = '08'

hour_0['hour'][23] = '09'

hour_0['index']=hour_0['hour']

hour_0 = hour_0.groupby('index').first()
plt.figure(figsize=(16,10), dpi= 80)

plt.plot('date', 'statistics', data=date_0, color='tab:red')



plt.ylim(13500, 22500)

xtick_location = date_0['date'].tolist()[::6]

plt.xticks(xtick_location,xtick_location, rotation=0, fontsize=12, horizontalalignment='center', alpha=.7)

plt.yticks(fontsize=12, alpha=.7)

plt.title("Traffic Accidents of NYX (2012 - 2019)", fontsize=20)

plt.grid(axis='both', alpha=.3)



plt.gca().spines["top"].set_alpha(0.0)    

plt.gca().spines["bottom"].set_alpha(0.3)

plt.gca().spines["right"].set_alpha(0.0)    

plt.gca().spines["left"].set_alpha(0.3)   

plt.show()
statistics_value = date_0['statistics'].values

doublediff = np.diff(np.sign(np.diff(statistics_value)))

peak_locations = np.where(doublediff == -2)[0] + 1



doublediff2 = np.diff(np.sign(np.diff(-1*statistics_value)))

trough_locations = np.where(doublediff2 == -2)[0] + 1





plt.figure(figsize=(16,10), dpi= 80)

plt.plot('date', 'statistics', data=date_0, color='tab:blue', label='Traffic Accident')

plt.scatter(date_0.date[peak_locations], date_0.statistics[peak_locations], marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')

plt.scatter(date_0.date[trough_locations], date_0.statistics[trough_locations], marker=mpl.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')





for t, p in zip(trough_locations[1::5], peak_locations[::3]):

    plt.text(date_0.date[p], date_0.statistics[p]+15, date_0.date[p], horizontalalignment='center', color='darkgreen')

    plt.text(date_0.date[t], date_0.statistics[t]-35, date_0.date[t], horizontalalignment='center', color='darkred')





plt.ylim(13500,24500)

xtick_location = date_0['date'].tolist()[::3]

xtick_labels = xtick_location

plt.xticks(xtick_location, xtick_labels, rotation=90, fontsize=12, alpha=.7)

plt.title("NY Traffic Accident (2012 - 2019)", fontsize=22)

plt.yticks(fontsize=12, alpha=.7)





plt.gca().spines["top"].set_alpha(.0)

plt.gca().spines["bottom"].set_alpha(.3)

plt.gca().spines["right"].set_alpha(.0)

plt.gca().spines["left"].set_alpha(.3)



plt.legend(loc='upper left')

plt.grid(axis='y', alpha=.3)

plt.show()
statistics_value = hour_0['statistics'].values

doublediff = np.diff(np.sign(np.diff(statistics_value)))

peak_locations = np.where(doublediff == -2)[0] + 1



doublediff2 = np.diff(np.sign(np.diff(-1*statistics_value)))

trough_locations = np.where(doublediff2 == -2)[0] + 1



# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

plt.plot('hour', 'statistics', data=hour_0, color='tab:blue', label='Traffic Accident')

plt.scatter(hour_0.hour[peak_locations], hour_0.statistics[peak_locations], marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')

plt.scatter(hour_0.hour[trough_locations], hour_0.statistics[trough_locations], marker=mpl.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')



# Annotate

for t, p in zip(trough_locations[1::5], peak_locations[::3]):

    plt.text(hour_0.hour[p], hour_0.statistics[p]+15, hour_0.hour[p], horizontalalignment='center', color='darkgreen')

    plt.text(hour_0.hour[t], hour_0.statistics[t]-35, hour_0.hour[t], horizontalalignment='center', color='darkred')



# Decoration

plt.ylim(10000,110000)

xtick_location = hour_0['hour'].tolist()[::1]

xtick_labels = xtick_location

plt.xticks(xtick_location, xtick_labels, rotation=90, fontsize=12, alpha=.7)

plt.title("NY Traffic Accident (hour)", fontsize=22)

plt.yticks(fontsize=12, alpha=.7)



# Lighten borders

plt.gca().spines["top"].set_alpha(.0)

plt.gca().spines["bottom"].set_alpha(.3)

plt.gca().spines["right"].set_alpha(.0)

plt.gca().spines["left"].set_alpha(.3)



plt.legend(loc='upper left')

plt.grid(axis='y', alpha=.3)

plt.show()
statistics_value = month_0['statistics'].values

doublediff = np.diff(np.sign(np.diff(statistics_value)))

peak_locations = np.where(doublediff == -2)[0] + 1



doublediff2 = np.diff(np.sign(np.diff(-1*statistics_value)))

trough_locations = np.where(doublediff2 == -2)[0] + 1



# Draw Plot

plt.figure(figsize=(16,10), dpi= 80)

plt.plot('month', 'statistics', data=month_0, color='tab:blue', label='Traffic Accident')

plt.scatter(month_0.month[peak_locations], month_0.statistics[peak_locations], marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')

plt.scatter(month_0.month[trough_locations], month_0.statistics[trough_locations], marker=mpl.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')



# Annotate

for t, p in zip(trough_locations[1::5], peak_locations[::3]):

    plt.text(month_0.month[p], month_0.statistics[p]+15, month_0.month[p], horizontalalignment='center', color='darkgreen')

    plt.text(month_0.month[t], month_0.statistics[t]-35, month_0.month[t], horizontalalignment='center', color='darkred')



# Decoration

plt.ylim(100000,140000)

xtick_location = month_0['month'].tolist()[::1]

xtick_labels = xtick_location

plt.xticks(xtick_location, xtick_labels, rotation=90, fontsize=12, alpha=.7)

plt.title("NY Traffic Accident (month)", fontsize=22)

plt.yticks(fontsize=12, alpha=.7)



# Lighten borders

plt.gca().spines["top"].set_alpha(.0)

plt.gca().spines["bottom"].set_alpha(.3)

plt.gca().spines["right"].set_alpha(.0)

plt.gca().spines["left"].set_alpha(.3)



plt.legend(loc='upper left')

plt.grid(axis='y', alpha=.3)

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

from dateutil.parser import parse



# Import Data

dates = pd.DatetimeIndex([parse(d).strftime('%Y-%m-01') for d in date_0['date']])

date_0.set_index(dates, inplace=True)



# Decompose 

result = seasonal_decompose(date_0['statistics'], model='multiplicative')



# Plot

plt.rcParams.update({'figure.figsize': (10,10)})

result.plot().suptitle('Time Series Decomposition of NY Traffic Accident')

plt.show()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(30,6), dpi= 80)

plot_acf(date_0.statistics.tolist(), ax=ax1, lags=70)

plot_pacf(date_0.statistics.tolist(), ax=ax2, lags=24)



# Decorate

# lighten the borders

ax1.spines["top"].set_alpha(.3); ax2.spines["top"].set_alpha(.3)

ax1.spines["bottom"].set_alpha(.3); ax2.spines["bottom"].set_alpha(.3)

ax1.spines["right"].set_alpha(.3); ax2.spines["right"].set_alpha(.3)

ax1.spines["left"].set_alpha(.3); ax2.spines["left"].set_alpha(.3)



# font size of tick labels

ax1.tick_params(axis='both', labelsize=12)

ax2.tick_params(axis='both', labelsize=12)

plt.show()
train=data[(data['LATITUDE']!='U')&(data['LONGITUDE']!='U')&(data['BOROUGH'].notnull())]

test=data[(data['LATITUDE']!='U')&(data['LONGITUDE']!='U')&(data['BOROUGH'].isnull())]

U=data[(data['LATITUDE']=='U')|(data['LONGITUDE']=='U')]
from datetime import date

from sklearn.ensemble import RandomForestClassifier



cols = ['LATITUDE', 'LONGITUDE']



reg_ngk = RandomForestClassifier(random_state=100)

reg_ngk.fit(train[cols], train['BOROUGH'])



test['BOROUGH_RF'] = reg_ngk.predict(test[cols])
test.head(3)
BOROUGHs = np.unique(test['BOROUGH_RF'])

colors = [plt.cm.tab10(i/float(len(BOROUGHs)-1)) for i in range(len(BOROUGHs))]



# Draw Plot for Each Category

plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')



for i, BOROUGH_RF in enumerate(BOROUGHs):

    plt.scatter('LONGITUDE', 'LATITUDE', 

                data=test.loc[test.BOROUGH_RF==BOROUGH_RF, :], 

                s=20, cmap=colors[i], label=str(BOROUGH_RF))

    # "c=" 修改为 "cmap="



# Decorations

plt.gca().set(xlim=(-74.28, -73.65), ylim=(40.48, 40.93),

              xlabel='LONGITUDE', ylabel='LATITUDE')



plt.xticks(fontsize=12); plt.yticks(fontsize=12)

plt.title("BOROUGH vs LONGITUDE/LATITUDE", fontsize=22)

plt.legend(fontsize=12)    

plt.show()    