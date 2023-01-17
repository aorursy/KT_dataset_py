import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import calendar
import folium
df2015 = pd.read_csv('/kaggle/input/daily-aqi-in-usa-counties-20152020/daily_aqi_by_county_2015.csv')
df2016 = pd.read_csv('/kaggle/input/daily-aqi-in-usa-counties-20152020/daily_aqi_by_county_2016.csv')
df2017 = pd.read_csv('/kaggle/input/daily-aqi-in-usa-counties-20152020/daily_aqi_by_county_2017.csv')
df2018 = pd.read_csv('/kaggle/input/daily-aqi-in-usa-counties-20152020/daily_aqi_by_county_2018.csv')
df2019 = pd.read_csv('/kaggle/input/daily-aqi-in-usa-counties-20152020/daily_aqi_by_county_2019.csv')
df2020 = pd.read_csv('/kaggle/input/daily-aqi-in-usa-counties-20152020/daily_aqi_by_county_2020.csv')
print(df2015.dtypes)
print('-'*40)
print(df2016.dtypes)
print('-'*40)
print(df2017.dtypes)
print('-'*40)
print(df2018.dtypes)
print('-'*40)
print(df2019.dtypes)
print('-'*40)
print(df2020.dtypes)
print('-'*40)
df = pd.concat([df2015, df2016, df2017, df2018, df2019, df2020], ignore_index=True)
df.head(3)
df.dtypes
states=['California', 'Texas', 'Florida', 'New York', 'Pennsylvania', 'Illinois', 'Ohio', 
        'Georgia', 'North Carolina', 'Michigan', 'New Jersey', 'Virginia', 'Washington', 
        'Arizona', 'Massachusetts', 'Tennessee', 'Indiana', 'Missouri', 'Maryland', 'Wisconsin', 
        'Colorado', 'Minnesota', 'South Carolina', 'Alabama', 'Louisiana', 'Kentucky', 'Oregon', 
        'Oklahoma', 'Connecticut', 'Iowa', 'Utah', 'Nevada', 'Arkansas', 'Mississippi', 'Kansas', 
        'New Mexico', 'Nebraska', 'West Virginia', 'Idaho', 'Hawaii', 'New Hampshire', 'Maine', 
        'Montana', 'Rhode Island', 'Delaware', 'South Dakota', 'North Dakota', 'Alaska', 
        'District Of Columbia', 'Vermont', 'Wyoming']
print('Exclude data from the following regions/countries:', 
      df[~df['State Name'].isin(states)]['State Name'].unique())
df = df[df['State Name'].isin(states)]
df.head()
df.Category.unique()
df_cat = df.groupby('Category')['AQI'].count().sort_values(ascending=False)
df_cat['Unhealthy'] = df_cat['Unhealthy']+df_cat['Very Unhealthy']+df_cat['Hazardous']
df_cat = df_cat[0:4]
df_cat
plt.figure(figsize=(12,8))
plt.pie(df_cat.values, colors=['#00db00', 'yellow', 'orange', 'red'], autopct='%1.1f%%',
       pctdistance=1.35, explode=(0., 0., 0.1, 0.15))
my_circle=plt.Circle( (0,0), 0.6, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('Air Quality (2015-2020)', fontsize=14, fontweight="bold")
plt.legend(['Good (AQI 0-50)', 'Moderate (AQI 51-100)', 'Unhealthy for Sensitive Groups (AQI 101-150)', 
            'Unhealthy (AQI>150)'], 
           bbox_to_anchor=(0.95, 0.85), loc='upper left')
plt.show()
good = len(df[df['AQI']<=50]['State Name'].unique())
moderate = len(df[(df['AQI']>=51) & (df['AQI']<=100)]['State Name'].unique())
unh_sens = len(df[(df['AQI']>=101) & (df['AQI']<=150)]['State Name'].unique())
[good, moderate, unh_sens]
set(states)-set(df[df['AQI']>=151]['State Name'].unique())
set(states)-set(df[df['AQI']>=201]['State Name'].unique())
sites = pd.read_csv('/kaggle/input/aqssites/aqs_sites.csv')
sites = sites[['State Code', 'County Code', 'Site Number', 'Latitude', 'Longitude', 'Elevation',
       'Site Established Date', 'Site Closed Date', 'State Name', 'County Name', 'City Name', 'CBSA Name']]
sites.head(3)
print('Exclude data from the following regions/countries:', 
      sites[~sites['State Name'].isin(states)]['State Name'].unique())
sites['Site Closed Date'] = pd.to_datetime(sites['Site Closed Date'])
cond = (( ( sites['Site Closed Date'].isna() ) | ( sites['Site Closed Date'].dt.year>=2015 ) )
        & ( sites['State Name'].isin(states) )
        & ( sites['Longitude']!=0 ) & ( sites['Latitude']!=0 )
       )
sites = sites[cond]
sites.head(3)
sites['Defining Site'] = sites['State Code']+'-'+sites['County Code'].astype(str).str.zfill(3)+'-'+sites['Site Number'].astype(str).str.zfill(4)
sites.head(3)
df_h = pd.merge(df[df['AQI']>300], sites, how='left', on='Defining Site')
m = folium.Map(location=[48, -102], zoom_start=3)

for i in df_h.index:
    if str(df_h['CBSA Name'][i])=='nan':
        t = df_h['State Name_x'][i]
    else:
        t = df_h['CBSA Name'][i]
    p = 'AQI='+str(df_h['AQI'][i])+' ('+str(df_h['Date'][i])+')'
    folium.Marker([df_h['Latitude'][i], df_h['Longitude'][i]], 
                  tooltip=t, popup=p,
                  icon=folium.Icon(icon='map-marker', color='darkred')).add_to(m)
m
site_max = df_h['Defining Site'].value_counts()
site_max_d = sites[sites['Defining Site']==site_max.index[0]]
print('In the period 2015-2020, there are ',df_h['Defining Site'].nunique(),
      ' sites with reported hazardous air quality.')
print('The most reported site is located in ',
      site_max_d['County Name'].values[0]+', '+site_max_d['State Name'].values[0],
      ' with ', site_max[0],' measurements of AQI>300.')
df_Minn = pd.merge(df[df['State Name'] == 'Minnesota'], sites, how='left', on='Defining Site')
df_Minn.Date = pd.to_datetime(df_Minn.Date)
print('In the period 2015-2020, there are ',df_Minn.shape[0], ' measurements over ',
      df_Minn.Date.nunique(), ' different dates.')
df_gr = df_Minn.groupby([df_Minn.Date.dt.year,'county Name','Category'])['AQI'].count().unstack(level=-1)
df_gr = df_gr[['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy']]
df_gr.fillna(0, inplace=True)
df_gr
fig, axes = plt.subplots(5, 1, figsize=(6,30))
for i in range(0,5):
    d = df_gr.xs(2015+i, level=0, axis=0).sort_values(['county Name'], ascending=False)
    d.plot.barh(ax=axes[i], stacked=True, color=['#00db00', 'yellow', 'orange', 'red'])
    axes[i].set_title(str(2015+i))
    axes[i].set_xlabel('Number of measurement days')
    axes[i].legend(bbox_to_anchor=(1.05, 1))
for ax in axes:
    ax.grid()
plt.show()
df_Henn = df[df['county Name'] == 'Hennepin'][['Date','AQI']]
df_Henn.Date = pd.to_datetime(df_Henn.Date)
df_Henn.head()
c = np.arange(1,6)
norm = matplotlib.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Blues)
cmap.set_array([])

fig, axes = plt.subplots(5, 1, figsize=(6,30))
for i in range(0,5):
    d = df_Henn[df_Henn.Date.dt.year==(2015+i)]    
    axes[i].plot(d['Date'], d['AQI'], c=cmap.to_rgba(i+2))
    axes[i].set_title(str(2015+i))
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('AQI')
for ax in axes:
    ax.grid()
plt.show()
df_Henn.set_index('Date', inplace=True)
def split_into_cycles():
    cycles = []
    for y in range(2015,2020):
        cycle = df_Henn[df_Henn.index.year.isin([y])]
        cycle.index = (cycle.index - cycle.index[0]).days
        cycle.reindex(pd.Int64Index(np.arange(0,366)))
        cycles.append(cycle)
    return cycles
aqi_cycles = split_into_cycles()

plt.figure(figsize=(15,8))
for i, cycle in enumerate(aqi_cycles):
    cycle['AQI'].plot(label=f"Year {i+2015}", c=cmap.to_rgba(i+2))    
plt.title('AQI levels for Hennepin county (2015-1019)', fontsize=16)
plt.xticks(np.arange(0, 366, step=30.5), calendar.month_name[1:13], rotation=20)
plt.xlabel('')
plt.ylabel('AQI')
plt.axhline(50, color='#00db00');
plt.axhline(100, color='yellow');
plt.legend()
plt.show()