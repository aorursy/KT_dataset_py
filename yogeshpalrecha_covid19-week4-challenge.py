# Load library

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from math import sqrt
# Read the data

train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
# Check data

print(train_df.shape)

train_df.tail()
print(test_df.shape)

test_df.head()
# Check for null values

train_df.isna().sum()
# Check for null values

test_df.isna().sum()
train_df['Province_State'].unique()
# Combining two data frame

all_data = pd.concat([train_df,test_df],axis=0,sort=False)

#all_data.tail()



# Fill Nan Values

all_data['Province_State'].fillna("None", inplace=True)

all_data['ConfirmedCases'].fillna(0, inplace=True)

all_data['Fatalities'].fillna(0, inplace=True)

all_data['Id'].fillna(-1, inplace=True)

all_data['ForecastId'].fillna(-1, inplace=True)

all_data.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



#all_data['Province_State'] = le.fit_transform(all_data['Province_State'])

#all_data['Country_Region'] = le.fit_transform(all_data['Country_Region'])



all_data['Date'] = pd.to_datetime(all_data['Date'])

all_data['Day_num'] = le.fit_transform(all_data.Date)

all_data['Day'] = all_data['Date'].dt.day

all_data['Month'] = all_data['Date'].dt.month

all_data['Year'] = all_data['Date'].dt.year



all_data.head()
# Create train and test data

train = all_data[all_data['ForecastId']==-1.0]

test = all_data[all_data['ForecastId']!=-1.0]
print(train.shape)

train.head()
# Total cases over the world 

temp = train.groupby('Date')['ConfirmedCases', 'Fatalities'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['ConfirmedCases', 'Fatalities'],

                 var_name='case', value_name='count')



fig = px.line(temp, x="Date", y="count", color='case',

              title='Total cases over the Date ', color_discrete_sequence = ['cyan', 'red'])

fig.show()
# Maximum confirmed and fatalities case on 2020-04-13

country_max = train.groupby(['Date','Country_Region'])['ConfirmedCases', 'Fatalities'].max().reset_index().sort_values(by='ConfirmedCases',ascending=False).groupby('Country_Region').max().reset_index().sort_values(by='ConfirmedCases',ascending=False)

country_max[:20].style.background_gradient(cmap='viridis_r')
# Getting Top country cases 

Top_country = train.groupby('Country_Region')['ConfirmedCases','Fatalities'].max().reset_index().sort_values(by='ConfirmedCases',ascending=False).head(15)



# confirmed - deaths

fig_c = px.bar(Top_country.sort_values('ConfirmedCases'), x="ConfirmedCases", y="Country_Region", 

               text='ConfirmedCases', orientation='h', color_discrete_sequence = ['cyan'])



fig_d = px.bar(Top_country.sort_values('Fatalities'), x="Fatalities", y="Country_Region", 

               text='Fatalities', orientation='h', color_discrete_sequence = ['red'])





fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,

                    subplot_titles=('Confirmedcases', 'Fatalities'))



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)

# Rise of Confirmed Cases around top 10 countries



countries = Top_country.Country_Region.unique().tolist()

df_plot = train.loc[(train.Country_Region.isin(countries[0:10])) & (train.Date >= '2020-03-01')][['Date', 'Country_Region', 'ConfirmedCases', 'Fatalities']].groupby(['Date', 'Country_Region']).max().reset_index()

df_plot = df_plot.groupby(['Date', 'Country_Region']).sum().sort_values(by='ConfirmedCases', ascending=False).reset_index()

#df_plot



fig = px.bar(df_plot, x="Date", y="ConfirmedCases", color="Country_Region", barmode="stack",)

fig.update_layout(title='Rise of Confirmed Cases around top 10 countries', annotations=[dict(x='2020-03-22', y=150, xref="x", yref="y", text="Corona Rise exponentially from here", showarrow=True, arrowhead=1, ax=-150, ay=-150)])

fig.show()

# Dsitribution over the world



formated_gdf = train.groupby(['Date', 'Country_Region'])['ConfirmedCases', 'Fatalities'].max().reset_index()

formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

#formated_gdf['size'] = formated_gdf['ConfirmedCases'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="Country_Region", locationmode='country names', 

                     color="ConfirmedCases", hover_name="Country_Region", 

                     range_color= [0, 1500], 

                     projection="natural earth", animation_frame="Date", 

                     title='COVID-19: Spread Over World', color_continuous_scale="portland")



fig.show()
# Apply label encoding

train['Province_State'] = le.fit_transform(train['Province_State'])

train['Country_Region'] = le.fit_transform(train['Country_Region'])



test['Province_State'] = le.fit_transform(test['Province_State'])

test['Country_Region'] = le.fit_transform(test['Country_Region'])



# Creating train data

X = train.drop(columns=['Id','ConfirmedCases','Fatalities','Date','ForecastId'],axis=1)

cases = train.ConfirmedCases

fatalities = train.Fatalities



x_test = test.drop(columns=['Id','ConfirmedCases','Fatalities','Date','ForecastId'],axis=1)
model = XGBRegressor(n_estimators = 1000 , random_state = 0 , max_depth = 15)

model.fit(X,cases)

cases_pred = model.predict(x_test)



model1 = XGBRegressor(n_estimators = 1000 , random_state = 0 , max_depth = 15)

model1.fit(X,fatalities)

fatalities_pred = model1.predict(x_test)

model
# Getting Accuracy value

MSE = mean_squared_error(cases.iloc[0:13459],cases_pred)

RMSE = sqrt(mean_squared_error(cases.iloc[0:13459],cases_pred))

MAE = mean_absolute_error(cases.iloc[0:13459],cases_pred)

R2 = r2_score(cases.iloc[0:13459],cases_pred)



print('Mean squared error :', MSE)

print('Root mean squared error :',RMSE)

print('Mean absolute error :', MAE)

print('R squared :',R2)
x_test.shape,test_df.shape
# Predicted Result

test_df_predict = test_df.copy()

test_df_predict['Confirmedcase'] = cases_pred

test_df_predict['Fatalities'] = fatalities_pred

test_df_predict = test_df_predict.drop('Province_State',axis=1)

test_df_predict.to_csv('Forecast_result.csv')
test_df_predict.head(15)
US = test_df_predict[test_df_predict['Country_Region']=='US']

US.groupby('Date')['Confirmedcase','Fatalities'].sum().reset_index()
# Forecasting Comparison by date

temp = test_df_predict.groupby('Date')['Confirmedcase', 'Fatalities'].max().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Confirmedcase', 'Fatalities'],

                 var_name='case', value_name='count')



fig = px.area(temp, x="Date", y="count", color='case',

              title='Forecasting cases over the Date ', color_discrete_sequence = ['cyan', 'red'])

fig.show()
test_df_predict.groupby(['Date','Country_Region'])['Confirmedcase','Fatalities'].max().reset_index().head(10)

                                                                                                    



#test_df_predict.sort_values(by='Confirmedcase',ascending=False).head(10)
test_df_predict.groupby('Country_Region')['Confirmedcase', 'Fatalities'].sum().reset_index().sort_values(by='Confirmedcase',ascending=False).head(15)

# Top 10 forecast result

top_country = test_df_predict.groupby('Country_Region')['Confirmedcase', 'Fatalities'].max().reset_index().sort_values(by='Confirmedcase',ascending=False).head(15)



# confirmed - Fatalities

fig_c = px.bar(top_country.sort_values('Confirmedcase'), x="Confirmedcase", y="Country_Region", 

               text='Confirmedcase', orientation='h', color_discrete_sequence = ['cyan'])



fig_d = px.bar(top_country.sort_values('Fatalities'), x="Fatalities", y="Country_Region", 

               text='Fatalities', orientation='h', color_discrete_sequence = ['red'])





fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,

                    subplot_titles=('Confirmedcase', 'Fatalities'),)



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)
countries = top_country.Country_Region.unique().tolist()

df_plot = test_df_predict.loc[(test_df_predict.Country_Region.isin(countries[0:10])) & (test_df_predict.Date >= '2020-04-02')] [['Date', 'Country_Region', 'Confirmedcase', 'Fatalities']].groupby(['Date', 'Country_Region']).max().reset_index()

df_plot = df_plot.groupby(['Date', 'Country_Region']).sum().sort_values(by='Confirmedcase', ascending=False).reset_index()

#df_plot



fig = px.bar(df_plot, x="Date", y="Confirmedcase", color="Country_Region", barmode="stack",)

fig.update_layout(title='Top 10 countries Confirmedcase')

fig.show()

countries = top_country.Country_Region.unique().tolist()

df_plot = test_df_predict.loc[(test_df_predict.Country_Region.isin(countries[0:10])) & (test_df_predict.Date >= '2020-04-02')] [['Date', 'Country_Region', 'Confirmedcase', 'Fatalities']].groupby(['Date', 'Country_Region']).max().reset_index()

df_plot = df_plot.groupby(['Date', 'Country_Region']).sum().reset_index()

#df_plot



fig = px.bar(df_plot, x="Date", y="Fatalities", color="Country_Region", barmode="stack")

fig.update_layout(title='Top 10 countries Fatalities')

fig.show()

# Appending result to submission file

cases_pred = [round(value) for value in cases_pred ]

fatalities_pred = [round(value) for value in fatalities_pred ]



submission['ConfirmedCases'] = cases_pred

submission['Fatalities'] = fatalities_pred

submission.to_csv('submission.csv',index=False)



submission.tail()