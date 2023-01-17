# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import datetime





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Defining the dataset

train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

sub=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')
#Unique data

country=pd.unique(train['Country/Region'])

print('Number of countries',len(country))

date=pd.unique(train['Date'])

print('number of days',len(date))
temp = train.groupby(['Country/Region', 'Province/State'])['ConfirmedCases', 'Fatalities'].max()

temp.head(20)
temp1=train.groupby('Date')['ConfirmedCases','Fatalities'].sum().reset_index()

temp1=temp1[temp1['Date']==max(temp1['Date'])].reset_index(drop=True)

temp1.style.background_gradient(cmap='Accent')
train_grouped = train.groupby('Country/Region')['ConfirmedCases', 'Fatalities'].sum().reset_index()
temp_f = train_grouped.sort_values(by='ConfirmedCases', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f.style.background_gradient(cmap='Reds')
# Confirmed and death in map



fig = px.choropleth(train_grouped, locations="Country/Region", 

                    locationmode='country names', color="ConfirmedCases", 

                    hover_name="Fatalities", range_color=[1,7000], 

                    color_continuous_scale="aggrnyl", 

                    title='Confirmed and Death case')

fig.update(layout_coloraxis_showscale=False)

fig.show()
formated_gdf = train.groupby(['Date', 'Country/Region'])['ConfirmedCases','Fatalities'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])

formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['ConfirmedCases'].pow(0.3)



fig2 = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 

                     color="ConfirmedCases", size='size', hover_name="Country/Region", 

                     range_color= [0, max(formated_gdf['ConfirmedCases'])+2], 

                     projection="natural earth", animation_frame="Date", 

                     title='Spread over time')

fig2.update(layout_coloraxis_showscale=False)

fig2.show()
locations = list(set([(test.loc[i, "Province/State"], test.loc[i, "Country/Region"]) for i in test.index]))

locations
train['Date'].min(),train['Date'].max()
test['Date'].max(),test['Date'].min()



submission = test[["ForecastId"]]

submission.insert(1, "ConfirmedCases", 0)

submission.insert(2, "Fatalities", 0)
#final evaluation

train_start_date = "2020-01-12"

last_train_date = "2020-03-22"

test_end_date  = "2020-03-26"



for loc in locations:

    if type(loc[0]) is float and np.isnan(loc[0]):

        confirmed=train[((train["Country/Region"]==loc[1]) & (train["Date"]==last_train_date))]["ConfirmedCases"].values[0]

        deaths=train[((train["Country/Region"]==loc[1]) & (train["Date"]==last_train_date))]["Fatalities"].values[0]

        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]<=test_end_date)), "ConfirmedCases"] = confirmed

        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]<=test_end_date)), "Fatalities"] = deaths

    else:

        confirmed=train[((train["Province/State"]==loc[0]) & (train["Country/Region"]==loc[1]) & (train["Date"]==last_train_date))]["ConfirmedCases"].values[0]

        deaths=train[((train["Province/State"]==loc[0]) & (train["Country/Region"]==loc[1]) & (train["Date"]==last_train_date))]["Fatalities"].values[0]

        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]<=test_end_date)), "ConfirmedCases"] = confirmed

        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]<=test_end_date)), "Fatalities"] = deaths



submission
last_train_date = max(train["Date"])



for loc in locations:

    if type(loc[0]) is float and np.isnan(loc[0]):

        confirmed=train[((train["Country/Region"]==loc[1]) & (train["Date"]==last_train_date))]["ConfirmedCases"].values[0]

        deaths=train[((train["Country/Region"]==loc[1]) & (train["Date"]==last_train_date))]["Fatalities"].values[0]

        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]>test_end_date)), "ConfirmedCases"] = confirmed

        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]>test_end_date)), "Fatalities"] = deaths

    else:

        confirmed=train[((train["Province/State"]==loc[0]) & (train["Country/Region"]==loc[1]) & (train["Date"]==last_train_date))]["ConfirmedCases"].values[0]

        deaths=train[((train["Province/State"]==loc[0]) & (train["Country/Region"]==loc[1]) & (train["Date"]==last_train_date))]["Fatalities"].values[0]

        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]>test_end_date)), "ConfirmedCases"] = confirmed

        submission.loc[((test["Country/Region"]==loc[1]) & (test["Date"]>test_end_date)), "Fatalities"] = deaths



submission
submission.to_csv("submission.csv", index=False)