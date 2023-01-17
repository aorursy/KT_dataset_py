import random
from datetime import timedelta


import numpy as np
import pandas as pd

# GÖRÜNTÜLEME
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import folium



# Çevirme
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()   

#Uyarıları gizleme
import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.

df=pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')
df['Date']=pd.to_datetime(df['Date'])
df.head(10)
Turkey_df=df[df['Country/Region']=='Turkey']
temp = df[df['Date'] == max(df['Date'])]

m = folium.Map(location=[0, 0], tiles='cartodbpositron',
               min_zoom=1, max_zoom=4, zoom_start=1)

for i in range(0, len(temp)):
    folium.Circle(
        location=[temp.iloc[i]['Lat'], temp.iloc[i]['Long']],
        color='Blue', fill='Blue',
        tooltip =   '<li><bold>Ulke : '+str(temp.iloc[i]['Country/Region'])+
                    '<li><bold>Bolge : '+str(temp.iloc[i]['Province/State'])+
                    '<li><bold>Onaylanmis : '+str(temp.iloc[i]['Confirmed'])+
                    '<li><bold>Death : '+str(temp.iloc[i]['Deaths']),
        radius=int(temp.iloc[i]['Confirmed'])**1.1).add_to(m)
m
#Corona virüsünün dünya üzerinde ki tarih ile yayılımı
figur = px.choropleth(df, locations="Country/Region", locationmode='country names', color=np.log(df["Confirmed"]), 
                    hover_name="Country/Region", animation_frame=df["Date"].dt.strftime('%Y-%m-%d'),
                    title='Cases over time', color_continuous_scale=px.colors.sequential.Magenta)
figur.update(layout_coloraxis_showscale=False)
figur.show()
country_wise = df[df['Date']==max(df['Date'])].reset_index(drop=True).drop('Date', axis=1)

# group by country
country_wise = country_wise.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

# per 100 cases
country_wise['Deaths / 100 Cases'] = round((country_wise['Deaths']/country_wise['Confirmed'])*100, 2)
country_wise['Recovered / 100 Cases'] = round((country_wise['Recovered']/country_wise['Confirmed'])*100, 2)
country_wise['Deaths / 100 Recovered'] = round((country_wise['Deaths']/country_wise['Recovered'])*100, 2)

cols = ['Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered']
country_wise[cols] = country_wise[cols].fillna(0)

display(country_wise.head())
display(country_wise[country_wise['Country/Region']=='Turkey'])
plt.figure(figsize=(16, 9))
plt.plot(Turkey_df['Date'], Turkey_df['Confirmed'])
plt.plot(Turkey_df['Date'], Turkey_df['Deaths'])
plt.plot(Turkey_df['Date'], Turkey_df['Recovered'])
plt.title('Turkiye Vaka Sayısı', size=20)
plt.xlabel('Ilk Günden Bugüne', size=20)
plt.ylabel('Cases', size=20)
plt.legend(['Vaka', 'Ölüm', 'İyileşen'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# confirmed - deaths
fig_c = px.bar(country_wise.sort_values('Confirmed').tail(15), x="Confirmed", y="Country/Region", 
               text='Confirmed', orientation='h', color_discrete_sequence = ['#ffff00'])
fig_d = px.bar(country_wise.sort_values('Deaths').tail(15), x="Deaths", y="Country/Region", 
               text='Deaths', orientation='h', color_discrete_sequence = ['#660099'])





# plot
fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,
                    subplot_titles=('Onaylanmis Vaka', 'Bildirilen Ölümler'))

fig.add_trace(fig_c['data'][0], row=1, col=1)
fig.add_trace(fig_d['data'][0], row=1, col=2)

fig.update_layout(height=500)

plt.figure(figsize=(10,8))
sns.scatterplot(x='Long', y= 'Lat', hue = "Deaths", data = df, alpha =0.2)
