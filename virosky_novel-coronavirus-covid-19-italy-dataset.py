import pandas as pd

import folium

import warnings

import seaborn as sns

import matplotlib.dates as md

import datetime

sns.set(style='darkgrid')

import plotly.express as px

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

import plotly

import numpy as np

import matplotlib.dates as mdates

import matplotlib.ticker as ticker

import ipywidgets as widgets

from ipywidgets import interact, interact_manual

import plotly.graph_objects as go

from scipy.optimize import curve_fit
covid19provlast = pd.read_csv("../input/italy-covid19/covid19-ita-province-latest.csv")

covid19provtime = pd.read_csv("../input/italy-covid19/covid19-ita-province.csv")

covid19regionlast= pd.read_csv("../input/italy-covid19/covid19-ita-regions-latest.csv")

covid19regiontime= pd.read_csv("../input/italy-covid19/covid19-ita-regions.csv")

ds_it=pd.read_csv('../input/italy-covid19/covid-nationality.csv')

covid_age=pd.read_csv('../input/italy-covid19/covid-age.csv')

covid_disease= pd.read_csv('../input/italy-covid19/covid-disease.csv')

covid19provtime.rename(columns={'ï»¿data': 'date'},inplace=True)
covid=covid19regionlast[['region','hospitalized_with_symptoms', 'intensive_care', 'total_hospitalized',

       'home_quarantine', 'total_confirmed_cases', 'new_confirmed_cases',

       'recovered', 'deaths', 'total_cases', 'swabs_made']]

covid.sort_values(by='total_confirmed_cases',ascending=False,inplace=True)

covid.style.background_gradient(cmap='BuGn')
totale=covid.sum().reset_index()

totale.drop([0,3,5,9,10,6],inplace=True)

totale.rename(columns={'index':'Detection',0 :'Total'},inplace=True)

totale.style.background_gradient(cmap='Blues')
labels=totale['Detection'].values.tolist()

sizes=totale['Total'].values.tolist()

explode = (0.1, 0 , 0.1, 0, 0)  

fig, ax = plt.subplots(figsize=(10,10))



ax.pie(sizes, explode=explode,labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.set_title('Cumulative Results',fontsize=20)

plt.tight_layout()

plt.show()
covid19prov=covid19provlast[['province','lat','long','total_cases']]

covid19prov['total_cases_q'] = pd.qcut(covid19prov['total_cases'], 3, labels=False)

covid19prov.fillna(0,inplace=True)



covid_map = folium.Map(location=[42.50, 12.50], zoom_start=5)



colordict = {0: 'green', 1: 'orange', 2: 'red', 3: 'purple'}



for lat,lon,total_q,total_cases,province in zip(covid19prov['lat'], covid19prov['long'], covid19prov['total_cases_q'],covid19prov['total_cases'],covid19prov['province']):

    folium.CircleMarker(

        [lat, lon],

        radius=.0030* int(total_cases),

        popup = ('City : ' + str(province).capitalize() + '<br>'

                 'Total cases : ' + str(int(total_cases)) + '<br>'),

        color='b',

        key_on = total_q,

        threshold_scale=[0,1,2,3],

        fill_color=colordict[total_q],

        fill=True,

        fill_opacity=0.47

        ).add_to(covid_map)

covid_map

italy_map = covid19provtime.groupby(['date', 'province'])['lat','long','total_cases'].max()



italy_map = italy_map.reset_index()

italy_map['size'] = italy_map['total_cases'].pow(0.5)

italy_map.head()



fig = px.scatter_mapbox(italy_map, lat="lat", lon="long",

                     color="total_cases", size='size', hover_name="province", hover_data=['total_cases'],

                     color_continuous_scale='matter',

                     animation_frame="date", 

                     title='Spread total cases over time in Italy')

fig.update(layout_coloraxis_showscale=True)

fig.update_layout(mapbox_style="carto-positron",

                  mapbox_zoom=4, mapbox_center = {"lat": 41.8719, "lon": 12.5674})

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

fig.show()
reg=px.bar(covid19regiontime,x='region', y="new_confirmed_cases", animation_frame="date", 

           animation_group="region", color="region", hover_name="region")

reg.update_yaxes(range=[0, 2500])

reg.update_layout(title='New Confirmed Cases')
tot=px.bar(covid19regiontime,x='region', y="total_cases", animation_frame="date", 

           animation_group="region", color="region", hover_name="region")

tot.update_yaxes(range=[0,50000])

tot.update_layout(title='Total cases')

tot.update_xaxes(categoryorder='total ascending')
ds_it['new_total_cases']=ds_it['total_cases']-ds_it['total_cases'].shift(1)

px.bar(ds_it,x='date',y='new_total_cases',title='New Total Cases',color_discrete_sequence=['cornflowerblue'])
new_confermed_total=ds_it[['date','new_confirmed_cases']]

regioni=covid19regiontime.loc[covid19regiontime.region.isin({'Lombardia','Veneto','Piemonte',

                                                             'Emilia-Romagna','Toscana','Marche','Liguria',

                                                             'Lazio','Campania','Puglia','Sicilia','P.A. Trento'})]

for i in regioni['region'].values:

    new_confermed_total[str(i)]= regioni.loc[regioni.region.isin({i})]['new_confirmed_cases'].values

    new_confermed_total[str(i) + "Percentage"]= round(new_confermed_total[i]/new_confermed_total['new_confirmed_cases'],2)
fig = go.Figure()

fig.add_trace(go.Scatter(x=new_confermed_total['date'], y=new_confermed_total['LombardiaPercentage'],

                    mode='lines',

                    name='Lombardia'))

fig.add_trace(go.Scatter(x=new_confermed_total['date'], y=new_confermed_total['Emilia-RomagnaPercentage'],

                    mode='lines+markers',

                    name='Emilia-Romagna'))

fig.add_trace(go.Scatter(x=new_confermed_total['date'], y=new_confermed_total['VenetoPercentage'],

                    mode='lines+markers',

                    name='Veneto'))

fig.add_trace(go.Scatter(x=new_confermed_total['date'], y=new_confermed_total['PiemontePercentage'],

                    mode='lines+markers',

                    name='Piemonte'))

fig.add_trace(go.Scatter(x=new_confermed_total['date'], y=new_confermed_total['MarchePercentage'],

                    mode='lines+markers',

                    name='Marche'))

fig.add_trace(go.Scatter(x=new_confermed_total['date'], y=new_confermed_total['ToscanaPercentage'],

                    mode='lines+markers',

                    name='Toscana'))

fig.add_trace(go.Scatter(x=new_confermed_total['date'], y=new_confermed_total['LiguriaPercentage'],

                    mode='lines+markers',

                    name='Liguria'))

fig.add_trace(go.Scatter(x=new_confermed_total['date'], y=new_confermed_total['LazioPercentage'],

                    mode='lines+markers',

                    name='Lazio'))

fig.add_trace(go.Scatter(x=new_confermed_total['date'], y=new_confermed_total['CampaniaPercentage'],

                    mode='lines+markers',

                    name='Campania'))

fig.update_layout(title='Region percentage of new confirmed cases')



fig.show()
lombardia=covid19regiontime.loc[covid19regiontime.region.isin({'Lombardia'})]

lombardia['new_total_cases']=lombardia['total_cases'] - lombardia['total_cases'].shift(1)

px.bar(lombardia,x='date',y='new_total_cases',title='New total cases in Lombardia',color_discrete_sequence=['orange'])
region_time_series=covid19regiontime.groupby('date')['deaths','total_cases'].sum().reset_index()

region_time_series['mortality_rate(%)']=round((region_time_series['deaths']/region_time_series['total_cases']),4)

px.line(region_time_series,x='date',y='mortality_rate(%)',title='Mortality rate(%) over time in Italy',color_discrete_sequence=['purple'])
temp = covid19regiontime.groupby('date')['recovered', 'deaths', 'total_confirmed_cases'].sum().reset_index()

temp = temp.melt(id_vars="date", value_vars=['recovered', 'deaths', 'total_confirmed_cases'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="date", y="Count", color='Case',

             title='Cases over time', color_discrete_sequence = ['orange','blue','red'])

fig.show()
sw=covid19regiontime.groupby(['date'])['new_confirmed_cases','swabs_made'].agg('sum').reset_index()

sw['swabs_per_day']=sw['swabs_made']- sw['swabs_made'].shift(1)

sw.drop('swabs_made',1,inplace=True)

sw['new_infection_per_day(%)']=round(sw['new_confirmed_cases']/sw['swabs_per_day'],2)



fig = go.Figure()

fig.add_trace(go.Bar(x=sw['date'],

                y=sw['swabs_per_day'],

                name='Tests Made every day',

                marker_color='rgb(55, 83, 109)'

                ))

fig.add_trace(go.Bar(x=sw['date'],

                y=sw['new_confirmed_cases'],

                name='New Confirmed Cases',

                marker_color='rgb(26, 118, 255)'

                ))



fig.update_layout(

    title='Tests Made and New confirmed Cases',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Number',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, 

    bargroupgap=0.1 

)

fig.show()
ds_it['hospitalized_everyday']=ds_it['hospitalized_with_symptoms']- ds_it['hospitalized_with_symptoms'].shift(1)

ds_it['intensive_care_everyday']= ds_it['intensive_care'] - ds_it['intensive_care'].shift(1)
fig = go.Figure()

fig.add_trace(go.Bar(x=ds_it['date'],

                y=ds_it['hospitalized_everyday'],

                name='Hospitalized with symptoms',

                marker_color='purple'

                ))

fig.add_trace(go.Bar(x=ds_it['date'],

                y=ds_it['intensive_care_everyday'],

                name='Intensive care',

                marker_color='orange'

                ))

fig.update_layout(title='Patients hospitalized and in intensive care every day')
covid_age.drop('Unnamed: 0',1,inplace=True)

covid_age['femal_mortality_rate']=round(covid_age['female_deaths']/covid_age['female_cases'],2)*100

covid_age['male_mortality_rate']=round(covid_age['male_deaths']/covid_age['male_cases'],2)*100

covid_age['mortality_rate']=round(covid_age['total_deaths']/covid_age['total_cases'],2)*100
fig,ax =plt.subplots(ncols=2,figsize=(20,7),dpi=100)

sns.barplot(x='age_classes',y='total_deaths',ax=ax[0],label='Female',data=covid_age,color='coral')

sns.barplot(x='age_classes',y='male_deaths',ax=ax[0],label='Male',data=covid_age,color='dodgerblue')

ax[0].legend()

ax[0].set_xlabel('Age Classes')

ax[0].set_title('Distribution of Deaths by age',fontsize=18)

ax[0].set_ylabel('Deaths')

sns.barplot(x='age_classes',y='total_cases',ax=ax[1],label='Female',data=covid_age,color='coral')

sns.barplot(x='age_classes',y='male_cases',ax=ax[1],label='Male',data=covid_age,color='dodgerblue')

ax[1].legend()

ax[1].set_xlabel('Age Classes')

ax[1].set_title('Distribution of Total Cases by age',fontsize=18)

ax[1].set_ylabel('Total Cases')

plt.show()
fig = go.Figure()

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['mortality_rate'],

                name='Lethality for Age Classes',

                marker_color='black'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['femal_mortality_rate'],

                name='Lethality for Female',

                marker_color='coral'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['male_mortality_rate'],

                name='Lethality for Male',

                marker_color='dodgerblue'

                ))

fig.update_layout(title='Lethality Rate (%)')

fig.update_yaxes(title='Lethality Rate (%)')

fig.update_xaxes(title='Age Classes')
fig = go.Figure()

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['lombardia_cases'],

                name='Lombardia',

                marker_color='blue'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['emilia-romagna_cases'],

                name='Emilia-Romagna',

                marker_color='red'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['veneto_cases'],

                name='Veneto',

                marker_color='green'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['piemonte_cases'],

                name='Piemonte',

                marker_color='black'

                ))



fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'},title='Most affected Regions')

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['valleaosta_cases'],

                name='Valle Aosta',

                marker_color='blue'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['friuli-veneziagiulia_cases'],

                name='Friuli-Venezia Giulia',

                marker_color='red'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['liguria_cases'],

                name='Liguria',

                marker_color='green'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['trento_cases'],

                name='Provincia Autonoma di Trento',

                marker_color='black'

                ))



fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'},title='Northern Regions')

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['toscana_cases'],

                name='Toscana',

                marker_color='blue'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['umbria_cases'],

                name='Umbria',

                marker_color='red'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['marche_cases'],

                name='Marche',

                marker_color='green'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['lazio_cases'],

                name='Lazio',

                marker_color='black'

                ))



fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'},title='Central Regions')

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['abruzzo_cases'],

                name='Abruzzo',

                marker_color='blue'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['molise_cases'],

                name='Molise',

                marker_color='red'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['campania_cases'],

                name='Campania',

                marker_color='green'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['puglia_cases'],

                name='Puglia',

                marker_color='black'

                ))

fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['sicilia_cases'],

                name='Sicilia',

                marker_color='yellow'

                ))



fig.add_trace(go.Bar(x=covid_age['age_classes'],

                y=covid_age['sardegna_cases'],

                name='Sardegna',

                marker_color='purple'

                ))





fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'},title='South Regions')

fig.show()
covid_dis=covid_disease.loc[0:10,]

fig = go.Figure()

fig.add_trace(go.Bar(y=covid_dis['Total'],

                x=covid_dis['Disease'],

                name='Total',

                marker_color='black'

                ))

fig.add_trace(go.Bar(y=covid_dis['Men'],

                x=covid_dis['Disease'],

                name='Men',

                marker_color='dodgerblue'

                ))

fig.add_trace(go.Bar(y=covid_dis['Women'],

                x=covid_dis['Disease'],

                name='Women',

                marker_color='coral'

                ))

fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'},title='Most common pre-existing chronic pathologies in patients who died')

fig.show()
covid_com=covid_disease.loc[11:,]

fig = go.Figure()

fig.add_trace(go.Bar(y=covid_com['Total'],

                x=covid_com['Disease'],

                name='Total',

                marker_color='black'

                ))

fig.add_trace(go.Bar(y=covid_com['Men'],

                x=covid_com['Disease'],

                name='Men',

                marker_color='dodgerblue'

                ))

fig.add_trace(go.Bar(y=covid_com['Women'],

                x=covid_com['Disease'],

                name='Women',

                marker_color='coral'

                ))

fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'},title='Number of Comorbidities observed in positive deceased patients')

fig.show()
print('Patients died with 0 pre-existing patologies: ' + str(round(covid_com.iloc[0,4]/covid_com['Total'].sum(),4)*100)+ ' %')

print('Patients died with 1 pre-existing patologies: ' + str(round(covid_com.iloc[1,4]/covid_com['Total'].sum(),4)*100)+ ' %')

print('Patients died with 2 pre-existing patologies: ' + str(round(covid_com.iloc[2,4]/covid_com['Total'].sum(),4)*100)+ ' %')

print('Patients died with 3 and over pre-existing patologies: ' + str(round(covid_com.iloc[3,4]/covid_com['Total'].sum(),4)*100)+ ' %')
# Shifts

ds_it['total_cases-1'] = ds_it.shift(periods=1, fill_value=0)['total_cases']

# Deltas

ds_it['total_cases_DELTA1'] = ds_it['total_cases'] - ds_it['total_cases-1']

# Shift of Deltas

ds_it['total_cases_DELTA1-1'] = ds_it.shift(periods=1)['total_cases_DELTA1']



try:

    ds_it['growth_factor_cum_infected'] = ds_it['total_cases_DELTA1'] / ds_it['total_cases_DELTA1-1']

except ZeroDivisionError:

    ds_it['growth_factor_cum_infected'] = 0
ds_it['date']=pd.to_datetime(ds_it['date'])

ds_it['date']=ds_it['date'].dt.date

ds_it.set_index('date',inplace=True)
ax = ds_it.plot(y='growth_factor_cum_infected',label='Growth Factor',figsize=(20,8),marker='o')

plt.axhline(y=1, color='red', linewidth=1, zorder=1, alpha=1, label='Inflection Point')

plt.axhline(y=0, color='green', linewidth=2, zorder=0, alpha=1, label='End of Epidemic')

ax.axvspan(pd.datetime(2020, 2, 24), pd.datetime(2020, 3, 11), facecolor='red', alpha=0.25)

plt.text(pd.datetime(2020, 3, 2),3,'No restrictions',fontsize=20,color='red')

plt.legend()



#ax.xaxis.set_minor_locator(mdates.DayLocator())

ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d\n%b\n%Y'))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b\n%Y'))

#ax.tick_params(axis='x', which='both', labelsize=12)

ax.xaxis.grid(True, which='both')

ax.axvline(pd.Timestamp('2020-03-11'),color='green')

ax.annotate('Total Lockdown',xy =(pd.Timestamp('2020-03-11'),2.5), xytext=(pd.Timestamp('2020-03-13'),3),

            arrowprops=dict(arrowstyle="simple"),fontsize=20,color='green')



ax.set_xlim([datetime.date(2020 ,2, 24), datetime.date(2020, 4, 25)])

plt.title('Growth Factor on Confirmed Cases Italy COVID-19',fontsize=20)



plt.show()