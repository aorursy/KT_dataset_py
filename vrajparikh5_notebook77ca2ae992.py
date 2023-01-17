
#pip install matplotlib seaborn ipywidgets pandas plotly==4.6.0 --upgrade --quiet
#!jupyter nbextension enable --py widgetsnbextension
#!jupyter nbextension enable --py widgetsnbextension --sys-prefix

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import ipywidgets as widgets
import json
import plotly.express as px
import numpy as np
%matplotlib inline
%matplotlib nbagg
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
slider = widgets.IntSlider(
    min=0,
    max=10,
    step=1,
    description='Slider:',
    value=3
)
display(slider)
amp = widgets.FloatSlider(min=1, max=10, value=1,description="Amp")
display(amp)
district_wise_crimes_df_raw = pd.read_csv("../input/district-wise-crimes-committed-ipc-2001-2012csv/District_wise_crimes_committed_IPC_2001_2012.csv")
district_wise_crimes_df = district_wise_crimes_df_raw.copy()
district_wise_crimes_df
#From the data we can analysis that rape = custodial rape + other rape
#So we can skip custodial rape and other rape and can directly use rape column

#lets look into other colums
list(district_wise_crimes_df.columns)
#From column names we can suspect sum columns are just sum of different columns
#Analysing colums THEFT, RAPE, KIDNAPPING AND ABDUCTION OF OTHERS
district_wise_crimes_df[['THEFT','AUTO THEFT','OTHER THEFT', 'KIDNAPPING & ABDUCTION',
 'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS','KIDNAPPING AND ABDUCTION OF OTHERS','RAPE',
 'CUSTODIAL RAPE','OTHER RAPE']]
#from above data we can conclude 
# THEFT = AUTO THEFT + OTHER THEFT
# KIDNAPPING & ABDUCTION =KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS + KIDNAPPING AND ABDUCTION OF OTHERS

#Lets analyze deeply into rape columns
district_wise_crimes_df['CUSTODIAL RAPE'].unique()
district_wise_crimes_df[district_wise_crimes_df['CUSTODIAL RAPE'] != 0][['RAPE','CUSTODIAL RAPE','OTHER RAPE']]
#Thus we can conclude RAPE = CUSTODIAL RAPE + OTHER RAPE

#Till now 
# THEFT = AUTO THEFT + OTHER THEFT
# KIDNAPPING & ABDUCTION =KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS + KIDNAPPING AND ABDUCTION OF OTHERS
#RAPE = CUSTODIAL RAPE + OTHER RAPE

''' 'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS'
    'DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
    'INSULT TO MODESTY OF WOMEN',
    'CRUELTY BY HUSBAND OR HIS RELATIVES',
    'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES',
    'RAPE'
'''
#Lets sum all these columns into one column 'CRIME AGAINST WOMEN'all these columns will be analyzed in depth as we move forward
district_wise_crimes_df['CRIME AGAINST WOMEN'] = district_wise_crimes_df['KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS']+district_wise_crimes_df['DOWRY DEATHS']+district_wise_crimes_df['ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY']+district_wise_crimes_df['CRUELTY BY HUSBAND OR HIS RELATIVES']+district_wise_crimes_df['RAPE']+district_wise_crimes_df['IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES']+district_wise_crimes_df['INSULT TO MODESTY OF WOMEN']
#So now we can avoid the below columns
merged_columns = [ 'AUTO THEFT','OTHER THEFT','KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS','KIDNAPPING & ABDUCTION',
    'DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
    'INSULT TO MODESTY OF WOMEN',
    'CRUELTY BY HUSBAND OR HIS RELATIVES',
    'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES','RAPE','CUSTODIAL RAPE','OTHER RAPE']
sate_wise_crimes_df = district_wise_crimes_df.groupby(['STATE/UT','YEAR']).sum().copy()
district_wise_crimes_df.shape
sate_wise_crimes_df.drop(columns=merged_columns, inplace=True)
sate_wise_crimes_df
col = list(sate_wise_crimes_df.columns)
col.remove("TOTAL IPC CRIMES")
col
#From the data we can analysis that rape = custodial rape + other rape
def variation_over_years(state_names,crime_categories,state,crime, fig_size):
    plt.figure(figsize=fig_size)
    plt.xticks(rotation=75)
    for state_name in state_names:
        IPC_CRIMES = sate_wise_crimes_df.loc[(state_name,)][crime_categories]
        plt.plot(IPC_CRIMES.index, IPC_CRIMES, 's-')
    if state:
        plt.legend(state_names)
    else :
        plt.legend(crime_categories)
    return IPC_CRIMES
total_crimes_all_yrs = district_wise_crimes_df.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().sort_values(ascending = False)


#.sort_values('TOTAL IPC CRIMES', ascending=False)
total_crimes_all_yrs
variation_over_years(list(total_crimes_all_yrs.head(10).index),['TOTAL IPC CRIMES'],True,False, (12,10));
col = list(sate_wise_crimes_df.sum().sort_values(ascending = False).index)
col.remove("TOTAL IPC CRIMES")
#as we dont have specific data regarding OTHER IPC CRIMES lets not consider it andd focus on specifics
col.remove("OTHER IPC CRIMES")
inidividual_toltal_of_crimes = sate_wise_crimes_df.sum()
inidividual_toltal_of_crimes[col]
plt.figure(figsize=(12,6))
plt.title("TOTAL CRIMES")
plt.ylabel('CRIMES');
plt.xlabel('NUMBER OF CRIMES (IN THOUSANDS)');
#plt.yticks(rotation=75)
sns.barplot(list(inidividual_toltal_of_crimes[col].values),col);
year_wise_crimes = district_wise_crimes_df.groupby('YEAR').sum().copy()
year_wise_crimes.drop(columns=merged_columns, inplace=True)
year_wise_crimes.drop(columns='TOTAL IPC CRIMES', inplace=True)
year_wise_crimes.drop(columns='OTHER IPC CRIMES', inplace=True)
cols = list(year_wise_crimes.columns)
for col_name in cols:
    year_wise_crimes[col_name] = year_wise_crimes[col_name].apply(lambda x : int(x/1000))
year_wise_crimes.shape
plt.figure(figsize=(12,6))
plt.title("YEAR-WISE CRIMES (IN THOUSANDS)")
sns.heatmap(year_wise_crimes, fmt="d", annot=True, cmap='Blues');
#LETS SEE WHICH CRIME IS HIGHEST IN WHICH STATE
only_state_wise_crimes = district_wise_crimes_df.groupby('STATE/UT').sum().copy()
only_state_wise_crimes.drop(columns=merged_columns, inplace=True)
only_state_wise_crimes.drop(columns='TOTAL IPC CRIMES', inplace=True)
only_state_wise_crimes.drop(columns='OTHER IPC CRIMES', inplace=True)
only_state_wise_crimes.drop(columns='YEAR', inplace=True)
cols = list(only_state_wise_crimes.columns)
for col_name in cols:
   only_state_wise_crimes[col_name] = only_state_wise_crimes[col_name].apply(lambda x : int(x/1000))
only_state_wise_crimes.shape
plt.figure(figsize=(12,10))
plt.title("STATE-WISE CRIMES (IN THOUSANDS)")
sns.heatmap(only_state_wise_crimes, fmt="d", annot=True, cmap='Blues');
#sns.heatmap(only_state_wise_crimes, fmt="d", annot=True);
#LETS SEE VARIATION IN TOP 10 CRIMES IN STATE HAVING HIGHEST CRIMES EVERY YEAR (MADHYA PRADESH)

col = list(sate_wise_crimes_df.sum().sort_values(ascending = False).head(12).index)
col.remove("TOTAL IPC CRIMES")
#as we dont have specific data regarding OTHER IPC CRIMES lets not consider it andd focus on specifics
col.remove("OTHER IPC CRIMES")
variation_over_years(['MADHYA PRADESH'],col,False,True, (18,10));
district_wise_crimes_df.groupby('STATE/UT').sum()
# india_states = json.load(open('states_india.geojson','r'))
# state_id_map = {}
# for feature in india_states['features']:
#     feature['id'] = feature['properties']['state_code']
#     state_id_map[feature['properties']['st_nm'].upper()] = feature['id']
# state_id_map
# df_geo_map = only_state_wise_crimes.copy()
# l= []
# c=0
# for x in df_geo_map.index:
#     if x in state_id_map:
#         c= c+1
#     else : 
#         l.append(x)
# l
# len(list(df_geo_map.index))
# l2= []
# c=0
# for x in df_geo_map.index:
#     if x in state_id_map:
#         l2.append(state_id_map[x])
#     elif x == 'A & N ISLANDS':
#         l2.append(35)
#     elif x == 'ARUNACHAL PRADESH':
#         l2.append(12)
#     elif x == 'D & N HAVELI':
#         l2.append(26)
#     elif x == 'DELHI UT':
#         l2.append(7)
#     else :
#         print(x)
        
# df_geo_map['id'] = l2
# df_geo_map
# fig = px.choropleth(df_geo_map, locations = 'id', 
#                     geojson=india_states, 
#                     color='THEFT',
#                     hover_data =['THEFT'] )
# fig.update_geos(fitbounds="locations", visible = False)
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
# fig.show()

from ipywidgets import Layout
dp = widgets.Dropdown(
    options=list(district_wise_crimes_df['STATE/UT'].unique()),
    value='MAHARASHTRA',
    description='STATE:',
    disabled=False,
    layout=Layout(display='inline-flex')
)

crime_list = list(sate_wise_crimes_df.columns)
crime_list.remove('TOTAL IPC CRIMES')
crime_list.remove('OTHER IPC CRIMES')

ds = widgets.SelectMultiple(
    options=crime_list,
    value=col,
    rows=4,
    description='CRIMES',
    disabled=False,
    layout=Layout(display='inline-flex')
)

def sq(state, crimes):
    crimes = list(crimes)
    variation_over_years([state],list(crimes),False,True, (18,10));


    
widgets.interact(sq, state=dp,crimes = ds);
# selected_crimes = widgets.SelectMultiple(
#     options=crime_list,
#     value=col,
#     rows=6,
#     description='CRIMES',
#     disabled=False,
#     layout=Layout(display='inline-flex')
# )

selected_crimes = widgets.Dropdown(
    options=crime_list,
    value='CRIME AGAINST WOMEN',
    description='CRIME',
    disabled=False,
    layout=Layout(display='inline-flex')
)

selected_states = widgets.SelectMultiple(
    options=list(district_wise_crimes_df['STATE/UT'].unique()),
    value=['MAHARASHTRA','MADHYA PRADESH', 'GOA', 'HARYANA','TAMIL NADU'],
    rows=3,
    description='STATES',
    disabled=False,
    layout=Layout(display='inline-flex')
)

# def display_comparison(states, crimes):
#     crimes = list(crimes)
#     states = list(states)
#     if(len(crimes) > 5):
#         print("YOU CAN SELECT ATMOST 5 CRIMES")
#         return
#     variation_over_years(states,crimes,True,False, (18,10));

def display_comparison(states, crimes):
    states = list(states)
    variation_over_years(states,[crimes],True,False, (18,10));
    
widgets.interact(display_comparison, states=selected_states,crimes = selected_crimes);
var = 'hi"/n'
var + 'bye'


