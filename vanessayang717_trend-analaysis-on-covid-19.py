import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import plotly.express as px
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
# Load the dataset
my_data= pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
countryContinent= pd.read_csv('/kaggle/input/countrycontinent/countryContinent.csv', encoding = "ISO-8859-1")



my_data["Last Update"] = pd.to_datetime(my_data["Last Update"])
my_data["ObservationDate"] = pd.to_datetime(my_data["ObservationDate"])
my_data['Country/Region'] = np.where(my_data['Country/Region'] == "Mainland China","China" ,  my_data['Country/Region']) 
my_data = my_data.rename(columns={"ObservationDate": "Date", "Country/Region": "Country"})
country_data = my_data.groupby(['Country','Date'])['Confirmed','Deaths','Recovered'].sum()
#country_data = country_data.set_index(['Country','Date'], inplace=True)
country_data.sort_index(inplace=True)
country_data['Country_New_Confirmed'] = np.nan 
country_data['Country_New_Deaths'] = np.nan 
country_data['Country_New_Recovered'] = np.nan 

for idx in country_data.index.levels[0]:
    country_data.Country_New_Confirmed[idx] = country_data.Confirmed[idx].diff()

for idx in country_data.index.levels[0]:
    country_data.Country_New_Deaths[idx] = country_data.Deaths[idx].diff()

for idx in country_data.index.levels[0]:
    country_data.Country_New_Recovered[idx] = country_data.Recovered[idx].diff()

country_data = country_data.reset_index()

# merge the data with continent information    
country_data = country_data.merge(countryContinent,  how='left', 
                            left_on='Country', 
                            right_on='Country',
                            suffixes=('','_right'))


Continent = country_data.groupby(["Date",'Continent'])['Confirmed'].sum().to_frame().reset_index()

Continent = Continent.pivot(index='Date', columns='Continent', values='Confirmed').reset_index()

Continent_percentage  = Continent.iloc[:,1:8].div(Continent.iloc[:,1:8].sum(axis=1), axis=0)
Continent_percentage= round(Continent_percentage,3) *100
Continent_percentage["Date"] =Continent["Date"]
# make the date the first column
cols = Continent_percentage.columns.tolist()
cols = cols[-1:] + cols[:-1]
Continent_percentage = Continent_percentage[cols]


Continent_percentage.tail(5)

last_day = country_data["Date"].max()
country_data[country_data["Date"] == last_day].sort_values("Country_New_Confirmed", ascending = False).iloc[:,[0,5,6,7]].head(10).reset_index(drop=True).style.background_gradient(cmap='Blues')
last_day = country_data["Date"].max()
country_data[country_data["Date"] == last_day].sort_values("Confirmed", ascending = False).iloc[:,[0,2,3,4]].head(10).reset_index(drop=True).style.background_gradient(cmap='Blues')
world_data = my_data.groupby('Date')['Confirmed','Deaths','Recovered'].sum()
world_data.reset_index(inplace=True)
world_data["Golbal_New_Confirmed"] = world_data["Confirmed"].diff()
world_data["Golbal_New_Deaths"] = world_data["Deaths"].diff()
world_data["Golbal_New_Recovered"] = world_data["Recovered"].diff()
world_data.tail(1)
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


world_data.plot(x='Date',y=['Confirmed','Deaths','Recovered'],kind='line',ax=ax1)
world_data.plot(x='Date',y=['Golbal_New_Confirmed','Golbal_New_Deaths','Golbal_New_Recovered'],kind='line', ax=ax2)
country_data = country_data.merge(world_data[['Date','Golbal_New_Confirmed','Golbal_New_Deaths','Golbal_New_Recovered']], how='inner', 
                            left_on='Date', 
                            right_on='Date',
                            suffixes=('','_world'))

country_data=country_data.sort_values(['Date', 'Country_New_Confirmed'], ascending=[True, False])
ranking = country_data.groupby("Date").head(5)
ranking['Country_New_Confirmed'] = ranking['Country_New_Confirmed'].astype(str)
ranking["Info"] = ranking['Country'].str.cat(ranking['Country_New_Confirmed'],sep=" : ")

plot_graph = ranking.groupby(['Date','Golbal_New_Confirmed'])['Info'].apply(list).to_frame()
plot_graph = plot_graph.reset_index()

fig = plt.figure(figsize=(5,5))
fig = px.line(plot_graph,x="Date", y="Golbal_New_Confirmed",hover_data=[ 'Info'])
fig.show()


fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
world_data.plot(x='Date',y=['Confirmed'],kind='line',ax=ax1)


Continent.plot(x='Date',y=['Asia', 'Northern America', 'Oceania', 'South America', 'Europe','Africa', 'Western Asia'],kind='line',ax=ax1)
#Continent_plot[["Asia","Date"].plot(x='Date',y=['Confirmed'],kind='line',ax=ax1)

country_data = my_data.groupby(['Country','Date'])['Confirmed','Deaths','Recovered'].sum()
#country_data = country_data.set_index(['Country','Date'], inplace=True)
country_data.sort_index(inplace=True)
country_data['Country_New_Confirmed'] = np.nan 
country_data['Country_New_Deaths'] = np.nan 
country_data['Country_New_Recovered'] = np.nan 

for idx in country_data.index.levels[0]:
    country_data.Country_New_Confirmed[idx] = country_data.Confirmed[idx].diff()

for idx in country_data.index.levels[0]:
    country_data.Country_New_Deaths[idx] = country_data.Deaths[idx].diff()

for idx in country_data.index.levels[0]:
    country_data.Country_New_Recovered[idx] = country_data.Recovered[idx].diff()

country_data = country_data.reset_index()

severe_country = country_data[country_data["Date"] == country_data['Date'].max()].sort_values('Confirmed', ascending = False).head(15)["Country"]
severe_country = severe_country.to_list()
severe_country.insert(0,"Date")

fig = plt.figure(figsize=(20,20))
fig = px.line(country_data[(country_data.Country.isin(severe_country)) & (country_data['Confirmed']>100)],  y="Confirmed", color='Country')
fig.update_layout(xaxis_rangeslider_visible=False, yaxis_type='log')
fig.update_layout(
    title= "US, Spain, and Italy is increasing faster than China was",
    xaxis_title="Day after 100th case",
    yaxis_title='Accumulated cases')
fig.show()
def exponential_rate(Country,q_date ='2020-1-1', lockdown = False):

    country = country_data[country_data["Country"] == Country]
    country = country[country["Confirmed"] >= 100]
    country["growing_ratio"] = country["Confirmed"].pct_change()+1
    country["growth_factor"] = country["Country_New_Confirmed"].pct_change()+1
    country['Five_days_avaerage_growth_factor'] =  country.loc[:,"growth_factor"].rolling(window=5,min_periods=2).mean()
    
    country["Day_after_100th"] = range(len(country))

    fig = plt.figure(figsize=(12,12))
    #fig, axs = plt.subplots(3, sharex=True, sharey=False)
    #fig.suptitle('Sharing both axes')
    #axs[0].plot(country['Date'], country['Confirmed'])
    #axs[1].plot(country['Date'], country['growing_ratio'])
    #axs[2].plot(country['Date'], country['Five_days_avaerage_growth_factor'])

    #
    ax1 = fig.add_subplot(411)
    country.plot(x="Date",y="Confirmed",kind='line',ax=ax1)
    
    ax2 = fig.add_subplot(412)
    country.plot(x="Date",y="Country_New_Confirmed",kind='line',ax=ax2)
   
    
    ax3 = fig.add_subplot(413)
    country.plot(x="Date",y="growing_ratio",kind='line',ax=ax3)
    plt.axhline(y = 1, color = "deepskyblue",linestyle = '--')
    
    ax4 = fig.add_subplot(414)
    #country.plot(x="Date",y="growth_factor",kind='line',ax=ax4)
    country.plot(x="Date",y="Five_days_avaerage_growth_factor",kind='line',ax=ax4)
    plt.axhline(y = 1, color = "deepskyblue",linestyle = '--')
    ax4.set_ylim([0,5])
    if lockdown:
        plt.axvline(x = q_date, color = "red")
    
       
    #return country[["Date","Day_after_100th","Confirmed","Country_New_Confirmed","growing_ratio","growth_factor"]]
    return 
exponential_rate("South Korea")
exponential_rate("Italy",'2020-03-09',lockdown = True)
exponential_rate("Spain",'2020-03-14',lockdown = True)
exponential_rate("UK",'2020-03-16',lockdown = True)
exponential_rate("France",'2020-03-17',lockdown = True)
exponential_rate("Germany", '2020-03-22',lockdown = True)
# '2020-03-18' is the date they locked down NY
exponential_rate("US", '2020-03-18',lockdown = True)
exponential_rate("Iran", '2020-01-01',lockdown = False)
