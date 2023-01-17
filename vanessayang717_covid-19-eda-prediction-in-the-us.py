import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')


# set template for state code and name matching 
states_2 = {
        'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AS': 'American Samoa', 'AZ': 'Arizona', 'CA': 'California',
        'CO': 'Colorado','CT': 'Connecticut','DC': 'District of Columbia', 'DE': 'Delaware','FL': 'Florida','GA': 'Georgia',
        'GU': 'Guam','HI': 'Hawaii','IA': 'Iowa','ID': 'Idaho','IL': 'Illinois','IN': 'Indiana','KS': 'Kansas','KY': 'Kentucky',
        'LA': 'Louisiana','MA': 'Massachusetts','MD': 'Maryland','ME': 'Maine','MI': 'Michigan','MN': 'Minnesota','MO': 'Missouri',
        'MP': 'Northern Mariana Islands','MS': 'Mississippi','MT': 'Montana','NA': 'National','NC': 'North Carolina','ND': 'North Dakota',
        'NE': 'Nebraska','NH': 'New Hampshire','NJ': 'New Jersey','NM': 'New Mexico','NV': 'Nevada','NY': 'New York','OH': 'Ohio',
        'OK': 'Oklahoma','OR': 'Oregon','PA': 'Pennsylvania','PR': 'Puerto Rico','RI': 'Rhode Island','SC': 'South Carolina',
        'SD': 'South Dakota','TN': 'Tennessee','TX': 'Texas','UT': 'Utah','VA': 'Virginia','VI': 'Virgin Islands','VT': 'Vermont',
        'WA': 'Washington', 'WI': 'Wisconsin','WV': 'West Virginia','WY': 'Wyoming'        
}
states = {y:x for x,y in states_2.items()}
# load data

my_data= pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
my_data["ObservationDate"] = pd.to_datetime(my_data["ObservationDate"])
my_data["ObservationDate"] = my_data["ObservationDate"].dt.date
my_data['Country/Region'] = np.where(my_data['Country/Region'] == "Mainland China","China" ,  my_data['Country/Region']) 
my_data = my_data.rename(columns={"ObservationDate": "Date", "Country/Region": "Country"})

# get us data
US_data = my_data[my_data["Country"] == "US"]
new = US_data["Province/State"].str.split(',', n=1, expand=True)
US_data["County"]= new[0] 
US_data["State"]= new[1] 
US_data["State"] = US_data["State"].str.strip()

# clean the US data
US_data['Code'] = US_data['County'].map(states)
US_data['State'] = np.where(US_data['State'].isnull(), US_data['Code'],US_data['State'])
US_data['State'] =  np.where(US_data['State'].isnull(),"Others",US_data['State'])
US_data['State'] =  np.where(US_data['County'] == 'Chicago','IL',US_data['State'])
US_data['Code'] =  np.where(US_data['County'] == 'Chicago','IL',US_data['State'])
US_data['State'] = US_data['State'].map(states_2)
US_data = US_data.drop(["SNo","Province/State","Country","Last Update"], axis = 1)

#load demographic data
state_demographic= pd.read_csv('/kaggle/input/usdemographicdataset/state_demographic_data.csv')
state_demographic['State'] = state_demographic['State'].str.strip()

#load flu data
flu_data = pd.read_csv('/kaggle/input/usdemographicdataset/Influenza_Pneumonia Mortality by State.csv')
state_data = US_data.groupby(['State','Date'])['Confirmed','Deaths','Recovered'].sum()
#state_data = state_data.set_index(['State','Date'], inplace=True)
state_data.sort_index(inplace=True)
state_data['State_New_Confirmed'] = np.nan 
state_data['State_New_Deaths'] = np.nan 
state_data['State_New_Recovered'] = np.nan
state_data['growing_ratio'] = np.nan 
state_data['growth_factor'] = np.nan 

for idx in state_data.index.levels[0]:
    state_data.State_New_Confirmed[idx] = state_data.Confirmed[idx].diff()

for idx in state_data.index.levels[0]:
    state_data.State_New_Deaths[idx] = state_data.Deaths[idx].diff()

for idx in state_data.index.levels[0]:
    state_data.State_New_Recovered[idx] = state_data.Recovered[idx].diff()
    
for idx in state_data.index.levels[0]:
    state_data.growing_ratio[idx] = round(state_data["Confirmed"].pct_change()+1,2)

for idx in state_data.index.levels[0]:
    state_data.growth_factor[idx] = round(state_data["State_New_Confirmed"].pct_change()+1,2)
    

state_data = state_data.reset_index()
state_data['Code'] = state_data['State'].map(states)
state_data['Five_days_avaerage_growth_factor'] =  state_data.loc[:,"growth_factor"].rolling(window=5,min_periods=2).mean()
today_data = state_data[state_data["Date"] == state_data["Date"].max()]
today_data.sort_values("State_New_Confirmed", ascending = False).iloc[:,[0,5,6,7]].head(10).reset_index(drop=True).style.background_gradient(cmap='Blues')
today_data.sort_values("Confirmed", ascending = False).iloc[:,[0,2,3,4]].head(10).reset_index(drop=True).style.background_gradient(cmap='Blues')
# examining the data through the perspective of demographic parameters 
today_geo_data = today_data.merge(state_demographic,  how='left', 
                            left_on='State', 
                            right_on='State',
                            suffixes=('','_right'))
today_geo_data['Infected_rate'] = round(today_geo_data['Confirmed']*1000 / today_geo_data["Population"],2)
today_geo_data['Death_rate'] = round(today_geo_data['Deaths']*1000 / today_geo_data["Population"],2)
# confirmed case 
fig = go.Figure(data=go.Choropleth(
    locations=today_data['Code'], # Spatial coordinates
    z = today_data['Confirmed'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Blues',
    colorbar_title = "Confirmed cases",
    #text = plot_df['text']
))

fig.update_layout(
    title_text = 'NY & NJ are severely hit',
    geo_scope='usa', # limite map scope to USA
)

fig.show()
# confirmed case 
fig = go.Figure(data=go.Choropleth(
    locations=today_geo_data['Code'], # Spatial coordinates
    z = today_geo_data['Infected_rate'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Blues',
    colorbar_title = "Infected_rate(pre 1000)",
    #text = plot_df['text']
))

fig.update_layout(
    title_text = 'Considering the size of population, Louisiana is facing a hard time too',
    geo_scope='usa', # limite map scope to USA
)

fig.show()
# Number of death
fig = go.Figure(data=go.Choropleth(
    locations=today_geo_data['Code'], # Spatial coordinates
    z = today_geo_data['Deaths'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Deaths",
    #text = plot_df['text']
))

fig.update_layout(
    title_text = 'NY has the highest death toll(4698) followed by NJ(1003)',
    geo_scope='usa', # limite map scope to USA
)

fig.show()
# Number of death
fig = go.Figure(data=go.Choropleth(
    locations=today_geo_data['Code'], # Spatial coordinates
    z = today_geo_data['Death_rate'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Death_rate(per 1000)",
    #text = plot_df['text']
))

fig.update_layout(
    title_text = 'Considering the size of population,  number of death in Louisiana is high as well',
    geo_scope='usa', # limite map scope to USA
)

fig.show()
# population density with infected rate 
fig = plt.figure(figsize=(3,3))
fig = px.scatter(today_geo_data, x = "Density", y = "Infected_rate", hover_name="State")
fig.show()
# Heath_expend with infected rate 
fig = plt.figure(figsize=(3,3))
fig = px.scatter(today_geo_data, x = "Heath_expend", y = "Death_rate", hover_name="State")
fig.show()
# Death_rate and elder population 
fig = plt.figure(figsize=(5,5))
fig = px.scatter(today_geo_data, x = "Over_65", y = "Death_rate", hover_name="State")
fig.show()
# compare with flu seasonal pattern 
geo_flu_data = today_geo_data.merge(flu_data,  how='left', 
                            left_on='Code', 
                            right_on='STATE',
                            suffixes=('','_right'))
geo_flu_data['RATE'] = geo_flu_data['RATE'] /100
fig = plt.figure(figsize=(3,3))
fig  = px.scatter( geo_flu_data, y = 'Infected_rate', x = 'RATE', color = 'YEAR',hover_name="State")
fig.show()
# set a function to capture the worst states
def severe_state(start,stop):
    severe = today_data.sort_values('Confirmed', ascending = False)["State"].to_list()
    severe.insert(0,"Date")
    severe = severe[start:stop+1]
    return severe
fig = plt.figure(figsize=(20,20))
fig = px.line(state_data[(state_data.State.isin(severe_state(1,10)) )& (state_data['Confirmed']>80)],  y="Confirmed", color='State')
fig.update_layout(xaxis_rangeslider_visible=False, yaxis_type='log')
#fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_layout(
    title= "US 10 most servere states",
    xaxis_title="Day after 100th case",
    yaxis_title='Accumulated cases')
fig.show()
fig = plt.figure(figsize=(20,20))
fig = px.line(state_data[(state_data.State.isin(severe_state(11,20)) )& (state_data['Confirmed']>80)],  y="Confirmed", color='State')
fig.update_layout(xaxis_rangeslider_visible=False, yaxis_type='log')
#fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_layout(
    title= "US 11-20 servere state",
    xaxis_title="Day after 100th case",
    yaxis_title='Accumulated cases')
fig.show()
fig = plt.figure(figsize=(20,20))
fig = px.line(state_data[(state_data.State.isin(severe_state(1,10)) )& (state_data['Confirmed']>80)],  y="growing_ratio", color='State')
#fig.update_layout(xaxis_rangeslider_visible=False, yaxis_type='log')
#fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_layout(
    title= "US",
    xaxis_title="Day after 100th case",
    yaxis_title='growing_ratio')
fig.show()

def exponential_rate(Code,q_date ='2020-1-1', lockdown = False):

    State = state_data[state_data["Code"] == Code]
    State = State[State["Confirmed"] >= 100]
    State['Five_days_avaerage_growth_factor'] =  State.loc[:,"growth_factor"].rolling(window=5,min_periods=2).mean()
    
    State["Day_after_100th"] = range(len(State))

    fig = plt.figure(figsize=(12,12))
    
    ax1 = fig.add_subplot(211)
    State.plot(x="Date",y="Confirmed",kind='line',ax=ax1)
    if lockdown:
        plt.axvline(x = q_date, color = "red")
    
        
    ax2 = fig.add_subplot(212)
    State.plot(x="Date",y=["State_New_Confirmed", "State_New_Deaths"],kind='bar',ax=ax2)
    plt.title(str(Code), fontsize=16)
    
     
    return State[["Date","Day_after_100th","Confirmed","State_New_Confirmed","State_New_Deaths","growing_ratio","growth_factor"]].tail(10)
# New York -> will last one more month 
exponential_rate('NY',q_date ='2020-03-22', lockdown = True)
# New Jersey -> Not clear 
exponential_rate('NJ',q_date ='2020-03-21', lockdown = True)
# Minnesota -> not clear
exponential_rate('MI',q_date ='2020-03-23', lockdown = True)
# Clifonia -> recent recorded-high peak 
exponential_rate('CA',q_date ='2020-03-24', lockdown = True)
#  louisiana-> just started 
exponential_rate('LA',q_date ='2020-03-23', lockdown = True)
# MA -> LONG WAY 
exponential_rate('MA',q_date ='2020-03-24', lockdown = True)

# Florida -> unsatble 
exponential_rate('FL',q_date ='2020-04-03', lockdown = True)
# SO LATE TOO LATE 
# Pennsylvania
exponential_rate('PA', q_date ='2020-04-21', lockdown = True)
# Illinois 
exponential_rate('IL',q_date ='2020-03-21', lockdown = True)
# Washington -> mild 
exponential_rate('WA',q_date ='2020-03-23', lockdown = True)
# WA -> mild but not stable ( already announced a month more )
exponential_rate('TX',q_date ='2020-03-23', lockdown = True)