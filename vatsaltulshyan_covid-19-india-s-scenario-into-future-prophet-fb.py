import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
import folium
import os
from fbprophet import Prophet
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
pd.set_option('display.max_rows',20000, 'display.max_columns',100)
df_corona_in_india = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
df_corona_india = pd.read_csv("../input/covid19-corona-virus-india-dataset/complete.csv")
#df_ICMR = pd.read_csv("../input/covid19-in-india/ICMRTestingDetails.csv")
df_Individual = pd.read_csv("../input/covid19-in-india/IndividualDetails.csv")
df_Hospital = pd.read_csv("../input/covid19-in-india/HospitalBedsIndia.csv")
df_Age = pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv")
df_Italy = pd.read_csv("../input/covid19-in-italy/covid19_italy_region.csv")
df_daily_US = pd.read_csv("../input/covid19-in-usa/us_states_covid19_daily.csv")
df_daily_USA = pd.read_csv("../input/covid19-in-usa/us_covid19_daily.csv")
df_lab = pd.read_csv("../input/covid19-in-india/ICMRTestingLabs.csv")
#df_utm_lab = pd.read_csv("../input/covid-center-utm/City_Lat_Lon.csv")
df_statewise = pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv")
df_population_of_india = pd.read_csv("../input/covid19-in-india/population_india_census2011.csv")
df_with_utm = pd.read_csv("../input/utm-of-india/UTM ZONES of INDIA.csv")
df_corona_in_india.head()
df_corona_in_india.info()
df_corona_in_india.fillna(0,inplace=True)
df_corona_in_india.isnull().sum()
df_corona_in_india['Deaths'].replace('0#',0,inplace=True)
# df_corona_in_india.drop(index=2192,in#place=True)
# df_corona_in_india.drop(index=2225,inplace=True)
df_corona_in_india['Deaths']=pd.to_numeric(df_corona_in_india['Deaths'])

#Total cases of corona in India
df_corona_in_india['Total Cases'] = df_corona_in_india['Cured'] + df_corona_in_india['Deaths'] + df_corona_in_india['Confirmed']
#Active cases of corona in India
df_corona_in_india['Active Cases'] = df_corona_in_india['Total Cases'] - df_corona_in_india['Cured'] - df_corona_in_india['Deaths']
df_corona_in_india.head()
#Till yesterday in India
import datetime 
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)
yesterday=yesterday.strftime('%d/%m/%y')
df1= df_corona_in_india[df_corona_in_india['Date']==yesterday]
fig = px.bar(df1, x='State/UnionTerritory', y='Total Cases', color='Total Cases', height=600)
fig.update_layout(
    title='Till {} Total Cases in India'.format(yesterday))
fig.show()
#Till yesterday Cases in India
df1= df_corona_in_india[df_corona_in_india['Date']==yesterday]
fig = px.bar(df1, x='State/UnionTerritory', y='Active Cases', color='Active Cases',barmode='group', height=600,color_continuous_scale=["red","blue"])
fig.update_layout(
    title='Till {} Active Cases in India'.format(yesterday))
fig.show()
df_corona_in_india['Date'] =pd.to_datetime(df_corona_in_india.Date,dayfirst=True)
#Daily Cases in India Datewise
corona_data = df_corona_in_india.groupby(['Date'])['Cured'].sum().reset_index().sort_values('Cured',ascending = True)

fig = px.bar(corona_data, y='Cured', x='Date',hover_data =['Cured'], color='Cured', height=600,color_continuous_scale=["blue","red"])
fig.update_layout(
    title='Cured Cases in India Datewise')
fig.show()
#Daily Cases in India Datewise
corona_data = df_corona_in_india.groupby(['Date'])['Total Cases'].sum().reset_index().sort_values('Total Cases',ascending = True)
corona_data['Daily Cases'] = corona_data['Total Cases'].sub(corona_data['Total Cases'].shift())
corona_data['Daily Cases'].iloc[0] = corona_data['Total Cases'].iloc[0]
corona_data['Daily Cases'] = corona_data['Daily Cases'].astype(int)
fig = px.bar(corona_data, y='Daily Cases', x='Date',hover_data =['Daily Cases'], color='Daily Cases', height=600,color_continuous_scale=["blue","red"])
fig.update_layout(
    title='Daily Cases in India Datewise')
fig.show()
#Total Cases in India Datewise
corona_data = df_corona_in_india.groupby(['Date'])['Total Cases'].sum().reset_index().sort_values('Total Cases',ascending = True)
fig = px.bar(corona_data, y='Total Cases', x='Date',hover_data =['Total Cases'], color='Total Cases', height=600,color_continuous_scale=["blue","red"])
fig.update_layout(title='Total Cases in India Datewise')
fig.show()
# pct_change is used to obtain the percentage change in consecutive rows.
corona_data['Corona Growth Rate'] = corona_data['Total Cases'].pct_change().mul(100).round(2)
#Corona Growth Rate Comparison with Previous Day
fig = px.bar(corona_data, y='Corona Growth Rate', x='Date',hover_data =['Corona Growth Rate','Total Cases'], height=600,color_continuous_scale=["blue","red"])
fig.update_layout(title='Corona Growth Rate(in Percentage) Comparison with Previous Day')
fig.show()
#Moratality Rate
corona_data = df_corona_in_india.groupby(['Date'])['Total Cases','Active Cases','Deaths'].sum().reset_index().sort_values('Date',ascending=False)
corona_data['Mortality Rate'] = ((corona_data['Deaths']/corona_data['Total Cases'])*100) #(Death/ total cases)*100
fig = go.Figure()
fig.add_trace(go.Scatter(x=corona_data['Date'], y=corona_data['Mortality Rate'],mode='lines+markers',name='Cases',marker_color='red'))
fig.update_layout(title_text='COVID-19 Mortality Rate in INDIA',plot_bgcolor='rgb(225,230,255)' ,xaxis_title="Date",
    yaxis_title="Mortality Rate")
fig.show()
#DAILY DEATHS IN INDIA
corona_data = df_corona_in_india.groupby(['Date'])['Deaths'].sum().reset_index().sort_values('Deaths',ascending = True)
corona_data['DAILY DEATHS'] = corona_data['Deaths'].sub(corona_data['Deaths'].shift())
corona_data['DAILY DEATHS'].iloc[0] = corona_data['Deaths'].iloc[0]
corona_data['DAILY DEATHS'] = corona_data['DAILY DEATHS'].astype(int)
fig = px.bar(corona_data, y='DAILY DEATHS', x='Date',hover_data =['DAILY DEATHS'], color='DAILY DEATHS', height=600,color_continuous_scale=["blue","red"])
fig.update_layout(
    title='DAILY DEATHS IN INDIA')
fig.show()
#DAILY DEATHS IN INDIA
corona_data = df_corona_in_india.groupby(['Date'])['Deaths'].sum().reset_index().sort_values('Deaths',ascending = True)
fig = px.bar(corona_data, y='Deaths', x='Date',hover_data =['Deaths'], color='Deaths', height=600,color_continuous_scale=["blue","red"])
fig.update_layout(
    title='DEATHS IN INDIA')
fig.show()
#Recovery Rate
corona_data = df_corona_in_india.groupby(['Date'])['Total Cases','Active Cases','Cured'].sum().reset_index().sort_values('Date',ascending=False)
corona_data['Recovery Rate'] = ((corona_data['Cured']/corona_data['Total Cases'])*100)
fig = go.Figure()
corona_data = corona_data[3:]
fig.add_trace(go.Scatter(x=corona_data['Date'], y=corona_data['Recovery Rate'],
                    mode='lines+markers',name='Cases',marker_color='green'))


fig.update_layout(title_text='COVID-19 Recovery Rate in INDIA',plot_bgcolor='rgb(225,230,255)',xaxis_title="Date",
    yaxis_title="Recovery Rate")
fig.show()
#DAILY Recovery IN INDIA
corona_data = df_corona_in_india.groupby(['Date'])['Cured'].sum().reset_index().sort_values('Cured',ascending = True)
corona_data['DAILY RECOVERY'] = corona_data['Cured'].sub(corona_data['Cured'].shift())
corona_data['DAILY RECOVERY'].iloc[0] = corona_data['Cured'].iloc[0]
corona_data['DAILY RECOVERY'] = corona_data['DAILY RECOVERY'].astype(int)
fig = px.bar(corona_data, y='DAILY RECOVERY', x='Date',hover_data =['DAILY RECOVERY'], color='DAILY RECOVERY', height=600,color_continuous_scale=["blue","red"])
fig.update_layout(
    title='DAILY RECOVERY IN INDIA')
fig.show()
#DAILY DEATHS IN INDIA
corona_data = df_corona_in_india.groupby(['Date'])['Cured'].sum().reset_index().sort_values('Cured',ascending = True)
fig = px.bar(corona_data, y='Cured', x='Date',hover_data =['Cured'], color='Cured', height=600,color_continuous_scale=["blue","red"])
fig.update_layout(
    title='TOTAL CURED IN INDIA')
fig.show()
#Total Cases in Indian States Datewise
corona_data = df_corona_in_india.groupby(['Date','State/UnionTerritory','Total Cases'])['Cured','Deaths','Active Cases'].sum().reset_index().sort_values('Total Cases',ascending = False)
fig = px.bar(corona_data, y='Total Cases', x='Date',hover_data =['State/UnionTerritory','Active Cases','Deaths','Cured'], color='Total Cases',barmode='group', height=600,color_continuous_scale=["blue","red"])
fig.update_layout(
    title='Indian States with Current Total Corona Cases')
fig.show()
#Pie chart visualization of states effected by coronavirus
fig = px.pie(df_Age, values='TotalCases', names='AgeGroup')
fig.update_layout(
    title='Age Group affected with COVID-19')
fig.show()
#Agewise Gender affected by COVID-19(There are lot of Nan Values)
fig = px.histogram(df_Individual.dropna(), x="age",color ='gender')
fig.update_layout(
    title='Agewise Gender affected by COVID-19')
fig.show()
#Agewise Covid-19 patients in State(There are lot of Nan Values)
fig = px.histogram(df_Individual.dropna(), x="age",color ='detected_state')
fig.update_layout(
    title='Agewise Covid-19 patients in State')
fig.show()
#Statewise Total Cases
df_new = df1.groupby(['State/UnionTerritory','Cured','Deaths','Active Cases'])['Total Cases'].sum().reset_index().sort_values('Total Cases',ascending = False)

#Using Merge two join two diffrent data frames and then sorting them in ascending
df_population_with_corona_case = df_population_of_india.merge(df_new, left_on='State / Union Territory', right_on='State/UnionTerritory')
df_population_with_corona_case=df_population_with_corona_case.drop(labels=['State/UnionTerritory','Sno'],axis=1)
df_population_with_corona_case=df_population_with_corona_case.sort_values('Total Cases',ascending=False)
df_population_with_corona_case
#Pie chart visualization of states effected by coronavirus
fig = px.pie(df_population_with_corona_case, values='Total Cases', names='State / Union Territory')
fig.update_layout(
    title='Pie chart visualization of states effected by coronavirus')
fig.show()
#Data of Population,Cases of corona and UTM 
df_pop_caro_utm = df_population_with_corona_case.merge(df_with_utm , left_on='State / Union Territory', right_on='State / Union Territory')
df_pop_caro_utm

df_Hospital.at[0,'State/UT'] = 'Andaman and Nicobar Islands'
df_hospital_beds = df_pop_caro_utm.merge(df_Hospital , left_on='State / Union Territory', right_on='State/UT')
df_hospital_beds['NumSubDistrictHospitals_HMIS'] = df_hospital_beds['NumSubDistrictHospitals_HMIS'].fillna(value=0.0)
df_hospital_beds = df_hospital_beds.drop(['Area','Density','Gender Ratio','Latitude','Longitude','Sno','State/UT'],axis=1)
df_hospital_beds['Total Beds'] = df_hospital_beds['NumRuralBeds_NHP18'] + df_hospital_beds['NumUrbanBeds_NHP18']
df_hospital_beds['Total Used Beds'] = df_hospital_beds['Active Cases']
df_hospital_beds['Total Unused Beds'] = df_hospital_beds['Total Beds'] - df_hospital_beds['Active Cases']
df_hospital_beds.head()
#Statewise COVID-19 Beds in INDIA
Beds = df_hospital_beds.groupby(['State / Union Territory'])['Total Beds','Total Unused Beds','Total Used Beds'].sum().reset_index().sort_values('State / Union Territory',ascending =True)

fig = go.Figure()
fig.add_trace(go.Bar(
    y=Beds['State / Union Territory'],
    x=Beds['Total Unused Beds'],
    name='Total Unused Beds',
    orientation='h',
        marker=dict(
        color='rgba(1000, 71, 80, 0.6)',
        line=dict(color='rgba(1000, 71, 80, 1.0)', width=3)
    )

))
fig.add_trace(go.Bar(
    y=Beds['State / Union Territory'],
    x=Beds['Total Used Beds'],
    name='Total Used Beds',
    orientation='h',
    marker=dict(
        color='rgba(100, 78, 139, 0.6)',
        line=dict(color='rgba(100, 78, 139, 1.0)', width=3)
    )    
))
fig.update_layout(
    title='BEDS for Covid19 Patients',xaxis_title="Count",
    yaxis_title="States")
fig.update_layout(barmode='stack')
fig.show()
fig = px.treemap(df_hospital_beds, path=['State / Union Territory'],hover_data=['State / Union Territory','Rural population','NumRuralHospitals_NHP18','NumRuralBeds_NHP18'], values='NumRuralHospitals_NHP18')
fig.update_layout(
    title='Statewise Rural Population and their Hospital Facilities for Covid19')
fig.show()
fig = px.treemap(df_hospital_beds, path=['State / Union Territory'],hover_data=['State / Union Territory','Urban population','NumUrbanHospitals_NHP18','NumUrbanBeds_NHP18'], values='NumUrbanHospitals_NHP18')
fig.update_layout(
    title='Statewise Urban Population and their Hospital Facilities for Covid19')
fig.show()
fig = px.bar(df_hospital_beds, x='State / Union Territory',y='Rural population',hover_data=['Rural population'])
fig.update_layout(
    title='Rural Population State Wise')
fig.show()
fig = px.bar(df_hospital_beds, x='State / Union Territory',y='NumRuralBeds_NHP18',hover_data=['NumRuralBeds_NHP18'])
fig.update_layout(
    title='Number of Rural Beds State wise')
fig.show()
fig = px.bar(df_hospital_beds, x='State / Union Territory',y='Urban population',hover_data=['Urban population'])
fig.update_layout(
    title='Urban Population State wise')
fig.show()
fig = px.bar(df_hospital_beds, x='State / Union Territory',y='NumUrbanBeds_NHP18',hover_data=['NumUrbanBeds_NHP18'])
fig.update_layout(
    title='Number of Urban Beds State wise')
fig.show()
#India's Map with Statewise data of Total Cases,Deaths and Cure
India_map = folium.Map(location=[20.5937, 78.9629],zoom_start=4.55)
fg=folium.FeatureGroup(name="my map")
fg.add_child(folium.GeoJson(data=(open('../input/states-of-india/states_of_india.json','r',encoding='utf-8-sig').read())))
India_map.add_child(fg)
for lat,lan,name,cured,deaths,cases in zip(df_pop_caro_utm['Latitude'],df_pop_caro_utm['Longitude'],df_pop_caro_utm['State / Union Territory'],df_pop_caro_utm['Cured'],df_pop_caro_utm['Deaths'],df_pop_caro_utm['Total Cases']):
    if(deaths == 0):
        folium.Marker(location=[lat,lan],popup="<b>State  : </b>"+name+ "<br> <b>Total Cases : </b> "+str(cases)+"<br> <b>Deaths : </b> "+str(deaths)+"<br> <b>Cured : </b> "+str(cured)).add_to(India_map)
    else:
        folium.Marker(location=[lat,lan],popup="<b>State  : </b>"+name+ "<br> <b>Total Cases : </b> "+str(cases)+"<br> <b>Deaths : </b> "+str(deaths)+"<br> <b>Cured : </b> "+str(cured),icon=folium.Icon(color="red")).add_to(India_map)
India_map
#Total Cases,Active Cases,Cured,Deaths from Corona Virus in India
corona_data = df_corona_in_india.groupby(['Date'])['Total Cases','Active Cases','Cured','Deaths'].sum().reset_index().sort_values('Date',ascending=False)
fig = go.Figure()
fig.add_trace(go.Scatter(x=corona_data['Date'], y=corona_data['Total Cases'],
                    mode='lines+markers',name='Total Cases'))
fig.add_trace(go.Scatter(x=corona_data['Date'], y=corona_data['Active Cases'], 
                mode='lines+markers',name='Active Cases'))
fig.add_trace(go.Scatter(x=corona_data['Date'], y=corona_data['Cured'], 
                mode='lines+markers',name='Cured'))
fig.add_trace(go.Scatter(x=corona_data['Date'], y=corona_data['Deaths'], 
                mode='lines+markers',name='Deaths'))
fig.update_layout(title_text='Curve Showing Different Cases from COVID-19 in India',plot_bgcolor='rgb(225,230,255)')
fig.show()
#ICMR APPROVED LABS
fig = px.histogram(df_lab, x="state",color ='type')
fig.update_layout(
    title='ICMR APPROVED LABS')
fig.show()
#Statewise COVID-19 Testing in INDIA
corona_data = df_statewise.groupby(['State'])['TotalSamples','Negative','Positive'].sum().reset_index().sort_values('State',ascending =True)
fig = go.Figure()
fig.add_trace(go.Bar(
    y=corona_data['State'],
    x=corona_data['Negative'],
    name='Negative',
    orientation='h',
        marker=dict(
        color='rgba(58, 71, 80, 0.6)',
        line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
    )

))
fig.add_trace(go.Bar(
    y=corona_data['State'],
    x=corona_data['Positive'],
    name='Positive',
    orientation='h',
    marker=dict(
        color='rgba(246, 78, 139, 0.6)',
        line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
    )    
))
fig.update_layout(
    title='Statewise COVID Tests',xaxis_title="Count",
    yaxis_title="States")
fig.update_layout(barmode='stack')
fig.show()
# Adding the period of lockdown in India as on 26/04/2020
holiday = pd.DataFrame(
                        {
                        'holiday' : 'LockDown',
                        'ds': pd.date_range(start="2020-03-24",end="2020-05-31"),
                        'lower_window': 0,
                        'upper_window': 0
                        }
)
#Forecasting of Total Cases for Next 30 Days
df = df_corona_in_india.groupby('Date')['Total Cases'].sum().reset_index()
# Assigining variables to dates and total cases(Target Class) 
df.columns = ['ds','y']
df['ds'] = pd.to_datetime(df['ds'])
# Prophet is a forcasting model made by Facebook
m = Prophet(holidays = holiday)
# Lets fit the model
m.fit(df)
# Getting the next 30 dates
future = m.make_future_dataframe(periods=30,include_history = False)
#Obtaining the forcast for the next 30 days
forecast = m.predict(future)
#Lets plot on the graph for a easy view and understanding
fig = go.Figure()
# yhat is the predicted value ds is the dates 
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                    mode='lines+markers',name='Cases',marker_color='Black'))
fig.update_layout(
    title='Forecasting of Total Cases in INDIA for Next 30 Days',xaxis_title="Date",
    yaxis_title="Count")
fig.show()
from fbprophet.diagnostics import cross_validation
# help(cross_validation)
df_cv = cross_validation(m, horizon='30 days', period='15 days', initial='1 days')
print(forecast)
m.plot(forecast)
m.plot_components(forecast)
from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
print(df_p)
from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')

#Forecasting of Deaths for Next 30 Days
df = df_corona_in_india.groupby('Date')['Deaths'].sum().reset_index()
df.columns = ['ds','y']
df['ds'] = pd.to_datetime(df['ds'])
#Removing the dates when no deaths were reported 
df = df[43:].reset_index().drop(['index'],axis=1)
m = Prophet(holidays = holiday )
m.fit(df)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                    mode='lines+markers',name='Cases',marker_color='red'))
fig.update_layout(
    title='Forecasting of Deaths in INDIA for Next 30 Days',xaxis_title="Date",
    yaxis_title="Count")
fig.show()
from fbprophet.diagnostics import cross_validation
# help(cross_validation)
df_cv = cross_validation(m, horizon='30 days', period='15 days', initial='1 days')
print(forecast)
m.plot(forecast)
m.plot_components(forecast)
from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
print(df_p)
from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')


#Total Corona Cases INDIA vs USA
corona_data_INDIA =df_corona_in_india.groupby(['Date'])['Total Cases'].sum().reset_index().sort_values('Date',ascending = True)
corona_data_USA= df_daily_US.groupby(['dateChecked'])['positive'].sum().reset_index().sort_values('dateChecked',ascending = True)
fig = go.Figure()

fig.add_trace(go.Scatter(x=corona_data_USA['dateChecked'], y=corona_data_USA['positive'],
                    mode='lines+markers',name='USA'))

fig.add_trace(go.Scatter(x=corona_data_INDIA['Date'], y=corona_data_INDIA['Total Cases'],
                    mode='lines+markers',name='India'))

fig.update_layout(title_text='Total Corona Cases INDIA vs USA',plot_bgcolor='rgb(225,230,255)',xaxis_title="Date",
    yaxis_title="Count")
fig.show()
#COVID-19 Recovered Cases in INDIA vs USA
corona_data_INDIA =df_corona_in_india.groupby(['Date'])['Cured'].sum().reset_index().sort_values('Date',ascending = True)
corona_data_USA= df_daily_US.groupby(['dateChecked'])['recovered'].sum().reset_index().sort_values('dateChecked',ascending = True)
fig = go.Figure()

fig.add_trace(go.Scatter(x=corona_data_USA['dateChecked'], y=corona_data_USA['recovered'],
                    mode='lines+markers',name='USA'))

fig.add_trace(go.Scatter(x=corona_data_INDIA['Date'], y=corona_data_INDIA['Cured'],
                    mode='lines+markers',name='India'))

fig.update_layout(title_text='COVID-19 Recovered Cases in INDIA vs USA',plot_bgcolor='rgb(225,230,255)',xaxis_title="Date",
    yaxis_title="Count")
fig.show()
#COVID-19 Death Cases in INDIA vs USA
corona_data_INDIA =df_corona_in_india.groupby(['Date'])['Deaths'].sum().reset_index().sort_values('Date',ascending = True)
corona_data_USA= df_daily_US.groupby(['dateChecked'])['death'].sum().reset_index().sort_values('dateChecked',ascending = True)
fig = go.Figure()

fig.add_trace(go.Scatter(x=corona_data_USA['dateChecked'], y=corona_data_USA['death'],
                    mode='lines+markers',name='USA'))

fig.add_trace(go.Scatter(x=corona_data_INDIA['Date'], y=corona_data_INDIA['Deaths'],
                    mode='lines+markers',name='India'))

fig.update_layout(title_text='COVID-19 Death Cases in INDIA vs USA',plot_bgcolor='rgb(225,230,255)',xaxis_title="Date",
    yaxis_title="Count")
fig.show()
#DAILY DEATHS IN USA
corona_data = df_daily_US.groupby(['dateChecked'])['death'].sum().reset_index().sort_values('death',ascending = True)
corona_data['DAILY DEATHS'] = corona_data['death'].sub(corona_data['death'].shift())
corona_data['DAILY DEATHS'].iloc[0] = corona_data['death'].iloc[0]
corona_data['DAILY DEATHS'] = corona_data['DAILY DEATHS'].astype(int)
fig = px.bar(corona_data, y='DAILY DEATHS', x='dateChecked',hover_data =['DAILY DEATHS'], color='DAILY DEATHS', height=500)
fig.update_layout(
    title='DAILY DEATHS IN USA')
fig.show()

