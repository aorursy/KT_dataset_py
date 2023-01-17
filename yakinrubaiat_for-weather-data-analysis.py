# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/Machine Learning Dataset/Temperature data.csv')
df.head()
from plotly.offline import init_notebook_mode, iplot

from plotly import graph_objs as go



# Initialize plotly

init_notebook_mode(connected=True)
def plotly_df(df, title=''):

    """Visualize all the dataframe columns as line plots."""

    common_kw = dict(x=df.index, mode='lines')

    data = [go.Scatter(y=df[c], name=c, **common_kw) for c in df.columns]

    layout = dict(title=title)

    fig = dict(data=data, layout=layout)

    iplot(fig, show_link=False)
mon_name = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
new_df = {'year':[],

          'month':[],

          'stations':[],

          'temp_max':[],

          'temp_min':[]

         }
for index, row in df.iterrows():

    #print("Break")

    for i,col_value in enumerate(row):

        #print(str(row[0])+" "+str(row[1])+" "+str(row[i+2])+" "+str(row[i+14]))

        #print(row[0])

        new_df['year'].append(row[0])

        new_df['month'].append(mon_name[i])

        new_df['stations'].append(row[1])

        new_df['temp_max'].append(row[i+2])

        new_df['temp_min'].append(row[i+14])

        if i==11:

            break
dumy = pd.DataFrame.from_dict(new_df)
dumy['months'] = pd.to_datetime(dumy['year'].astype(str)  + dumy['month'], format='%Y%B')
df_main = dumy.drop(['year','month'],axis=1)
df_main.head()
mon = {

    'year':[],

    'month':[]

}

for year in range(2020,2041):

    for month in mon_name:

        #print(year)

        mon['year'].append(str(year))

        mon['month'].append(str(month))
future = pd.DataFrame.from_dict(mon)
future['ds'] = pd.to_datetime(future['year'].astype(str)  + future['month'], format='%Y%B')
future = future.drop(['year','month'],axis=1)
future.head()
from fbprophet import Prophet



import logging

logging.getLogger().setLevel(logging.ERROR)
def inverse_boxcox(y, lambda_):

    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)
def train_forecast(train_df):

    train_df2 = train_df.copy().set_index('ds')

    train_df2['y'], lambda_prophet = stats.boxcox(train_df2['y'])

    train_df2.reset_index(inplace=True)

    

    m = Prophet(seasonality_mode='multiplicative')

    m.fit(train_df2)

    forecast = m.predict(future)

    

    forecast['yhat'] = inverse_boxcox(forecast['yhat'], lambda_prophet)

    

    return forecast[['ds','yhat']]
data_all = []

for station in df_main['stations'].unique():

    df = df_main[df_main['stations']==station]



    df_max = df[['months','temp_max']]

    df_max.columns = ['ds', 'y']

    for_max = train_forecast(df_max)

    for_max.columns = ['ds', 'y_tem_max']



    df_min = df[['months','temp_min']]

    df_min.columns = ['ds', 'y']

    for_min = train_forecast(df_min)

    for_min.columns = ['ds', 'y_tem_min']



    for_main_temp = pd.merge(left=for_max, right=for_min, how='inner')

    for_main_temp['stations'] = station

    

    data_all.append(for_main_temp)
data_all = pd.concat(data_all, ignore_index=False)
data_all['months'] = data_all['ds'].astype(str)
data_all = data_all.drop(['ds'],axis=1)
data_all.to_excel('output_temp.xlsx',index=False,encoding='utf16')
rain_file_path = '/kaggle/input/Machine Learning Dataset/Rainfal.Test data.csv'
df_rain = pd.read_csv(rain_file_path)
df_rain.head()
days = [i for i in range(1,32)]
df_rain.columns
new_df = {'stations':[],

          'year':[],

          'month':[],

          'day':[],

          'rain':[]

         }
import datetime



def check_date(year, month, day):

    correctDate = None

    try:

        newDate = datetime.datetime(year, month, day)

        correctDate = True

    except ValueError:

        correctDate = False

    return correctDate
for index, row in df_rain.iterrows():

    for i,col_value in enumerate(row):

        if check_date(row[1],row[2],days[i])==True:

            new_df['stations'].append(row[0])

            new_df['year'].append(row[1])

            new_df['month'].append(mon_name[row[2]-1])

            new_df['day'].append(days[i])

            if type(row[i+3])==int:

                new_df['rain'].append(row[i+3])

            else:

                new_df['rain'].append(2)

        if i==30:

            break
print(len(new_df['stations']),len(new_df['rain']))
dumy = pd.DataFrame.from_dict(new_df)
dumy.head()
dumy['days'] = pd.to_datetime(dumy['year'].astype(str)  + dumy['month'] + dumy['day'].astype(str) , format='%Y%B%d',dayfirst=True)
df_rain = dumy.drop(['year','month','day'],axis=1)
df_rain.head()
df_rain = df_rain.fillna(0)
df_rain.head()
df_rain['rain'] = df_rain['rain'].astype(int)
df_rain.head()
months = [i for i in range(1,13)]
mon = {

    'year':[],

    'month':[],

    'day':[]

}

for year in range(2020,2041):

    for i,month in enumerate(months):

        for day in days:

            if check_date(year,month,day):

                mon['year'].append(str(year))

                mon['month'].append(mon_name[i])

                mon['day'].append(str(day))
future = pd.DataFrame.from_dict(mon)
future
future['ds'] = pd.to_datetime(future['year'].astype(str)  + future['month'] + future['day'].astype(str), format='%Y%B%d')

future = future.drop(['year','month','day'],axis=1)
future.head()
def train_forecast(train_df,future):

    train_df2 = train_df.copy().set_index('ds')

    #train_df2['y'], lambda_prophet = stats.boxcox(train_df2['y'])

    train_df2.reset_index(inplace=True)

    

    m = Prophet(seasonality_mode='multiplicative')

    m.fit(train_df2)

    forecast = m.predict(future)

    

    #forecast['yhat'] = inverse_boxcox(forecast['yhat'], lambda_prophet)

    

    return forecast[['ds','yhat']]
df_rain
data_all = []

for station in df_rain['stations'].unique():

    df = df_rain[df_rain['stations']==station]



    df_mn = df[['days','rain']]

    df_mn.columns = ['ds', 'y']

    for_main = train_forecast(df_mn,future)

    for_main.columns = ['ds', 'y_rain']



    for_main['stations'] = station

    

    data_all.append(for_main)
data_all = pd.concat(data_all, ignore_index=False)
data_all.head(50)
data_all['days'] = data_all['ds'].astype(str)

data_all = data_all.drop(['ds'],axis=1)
data_all.to_excel('output_rain.xlsx',index=False,encoding='utf16')
df_run= pd.read_csv('/kaggle/input/Machine Learning Dataset/runoff.csv')
df_run.columns
new_df = {'year':[],

          'month':[],

          'stations':[],

          'run_off':[]

         }
for index, row in df.iterrows():

    #print("Break")

    for i,col_value in enumerate(row):

        #print(str(row[0])+" "+str(row[1])+" "+str(row[i+2])+" "+str(row[i+14]))

        #print(row[0])

        new_df['year'].append(row[0])

        new_df['month'].append(mon_name[i])

        new_df['stations'].append(row[1])

        new_df['run_off'].append(row[i+2])

        if i==11:

            break
dumy = pd.DataFrame.from_dict(new_df)
dumy['months'] = pd.to_datetime(dumy['year'].astype(str)  + dumy['month'], format='%Y%B')
dumy.head()
df_main = dumy.drop(['year','month'],axis=1)
df_main.head()
mon = {

    'year':[],

    'month':[]

}

for year in range(2020,2041):

    for month in mon_name:

        #print(year)

        mon['year'].append(str(year))

        mon['month'].append(str(month))
future = pd.DataFrame.from_dict(mon)
future['ds'] = pd.to_datetime(future['year'].astype(str)  + future['month'], format='%Y%B')
future = future.drop(['year','month'],axis=1)
future.tail()
def train_forecast(train_df):

    train_df2 = train_df.copy().set_index('ds')

    train_df2['y'], lambda_prophet = stats.boxcox(train_df2['y'])

    train_df2.reset_index(inplace=True)

    

    m = Prophet(seasonality_mode='multiplicative')

    m.fit(train_df2)

    forecast = m.predict(future)

    

    forecast['yhat'] = inverse_boxcox(forecast['yhat'], lambda_prophet)

    

    return forecast[['ds','yhat']]
data_all = []

for station in df_main['stations'].unique():

    df = df_main[df_main['stations']==station]



    df_run = df[['months','run_off']]

    df_run.columns = ['ds', 'y']

    for_run = train_forecast(df_run)

    for_run.columns = ['ds', 'y_run_off']



    for_run['stations'] = station

    

    data_all.append(for_run)
data_all = pd.concat(data_all, ignore_index=False)
data_all['months'] = data_all['ds'].astype(str)
data_all = data_all.drop(['ds'],axis=1)
data_all.tail()
data_all.to_excel('output_runoff.xlsx',index=False,encoding='utf16')