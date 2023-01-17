# data analysis and wrangling
import numpy as np
import pandas as pd
import nltk
from wordcloud import WordCloud, STOPWORDS
from PIL import Image


# visualization
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
from matplotlib import animation, rc
%matplotlib notebook
 
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import os
import io
import base64
from IPython.display import HTML, display
from scipy.misc import imread
import codecs
from subprocess import check_output
print(os.listdir("../input/"))
print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1')
data.head()
data = pd.DataFrame(data[['iyear', 'imonth', 'iday', 'country_txt', 'region_txt', 'city', 'latitude', 'longitude', 'attacktype1_txt',
            'target1', 'targtype1_txt', 'gname', 'weaptype1_txt', 'success', 'nkill', 'nwound', 'summary', 'motive']])
data.head()
data.rename(columns={'iyear': 'year', 'imonth': 'month', 'iday': 'day', 'country_txt': 'country', 'region_txt': 'region',
                    'attacktype1_txt': 'attackType', 'target1': 'target', 'targtype1_txt': 'targetType', 'gname': 'groupName',
                    'weaptype1_txt': 'weaponType', 'nkill': 'killed', 'nwound': 'wounded'}, inplace=True)
data.head()
plt.figure(figsize=(14,7))
plt.tight_layout()
sns.countplot('year', data=data)
plt.xticks(rotation=90)
plt.title('Number of Terrorist Activities Each Year')
plt.show()
print('Top 10 terrorism affected countries:')
print(data['country'].value_counts().head(10))
plt.figure(figsize=(14,7))
sns.barplot(x= data['country'].value_counts()[:10].index, y=data['country'].value_counts()[:10].values, palette='rocket')
plt.title('Top 10  Terrorism Affected Countries')
plt.show()
print('Top regions in terms of terrorism:')
data['region'].value_counts().head(10)
plt.figure(figsize=(14,7))
sns.countplot(x= 'region', data=data, palette='rocket', order=data['region'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Top 10 Terrorism Affected Regions')
plt.show()
plt.figure(figsize=(14,7))
plt.tight_layout()
sns.countplot('attackType', data=data, order=data['attackType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attack methods used by terrorists')
plt.show()
plt.figure(figsize=(17, 7))
plt.tight_layout()
sns.countplot(data['targetType'], order=data['targetType'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Favourite targets of terrorist')
plt.show()
data.head()
#extract the data we are interested in

lat = data['latitude'].values
long = data['longitude'].values
fig = plt.figure(figsize=(20,10))
m = Basemap(projection='mill', resolution='c', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)

m.shadedrelief()
m.drawcoastlines()
m.drawcountries()
m.scatter(long, lat, latlon=True, c='r', alpha=0.4, s=3)
plt.show()
fig = plt.figure(figsize=(20,10))
m1 = Basemap(projection='mill', resolution='c', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, lat_ts=20,
           lat_0=True, lat_1=True)

m1.shadedrelief()
m1.drawcoastlines()
m1.drawcountries()

color_list = ['#f97704','#324732', '#cb9d1d', '#cd3333','#1a1a1a','#ff753b','#490c66','#ea1d75','#00bbb3','#8b0000',
             '#440567','#043145','#da785b','#e00062']

group_list = data[data['groupName'].isin(data['groupName'].value_counts()[:14].index)]
regional_wise_active = list(group_list['groupName'].unique())

def draw_map(groupName, color, label):
    lat_g = list(group_list[group_list['groupName']==groupName].latitude)
    lon_g = list(group_list[group_list['groupName']==groupName].longitude)
    x, y = m1(lon_g, lat_g)
    m1.scatter(x, y, c=j , label=i, alpha=0.8, s=3)
    
for i, j in zip(regional_wise_active, color_list):
    draw_map(i, j, i)
    
leg = plt.legend(loc='lower left', frameon=True, prop={'size':10})
frame = leg.get_frame()
frame.set_facecolor('white')
plt.title('Most Active Group in Particular Regions')
plt.show()
data_region = pd.crosstab(data.year, data.region)
data_region.plot(figsize=(18,6))
plt.show()
top10Groups = data[data['groupName'].isin(data['groupName'].value_counts()[1:15].index)]
pd.crosstab(top10Groups.year, top10Groups.groupName).plot(figsize=(18, 6))
region_attack = pd.crosstab(data.region, data.attackType)
region_attack.plot.bar(stacked=True, colormap='magma', figsize=(16, 8))
plt.show()
num_of_attacks = data['country'].value_counts()[:10].to_frame()
num_of_attacks.columns=['attacks']
people_killed = data.groupby('country')['killed'].sum().to_frame()
num_of_attacks.merge(people_killed, left_index=True, right_index=True, how='left').plot.bar(width=0.8, figsize=(18,6))
plt.show()
fig = plt.figure(figsize=(14, 8))
sns.barplot(data['groupName'].value_counts()[1:15].values, data['groupName'].value_counts()[1:15].index, palette=('hot'))
plt.xticks(rotation=90)
plt.title('Most Notorious Groups')
plt.show()
data['casualties'] = data['killed'] + data['wounded']
fig = plt.figure(figsize=(19,10))
def animate(year):
    ax = plt.axes()
    ax.clear()
    ax.set_title('Terrorism in world' + '\n' + str(year))
    m2 = Basemap(projection='mill', resolution='c', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)
    lat_ani = list(data[data['year']==year].latitude)
    long_ani = list(data[data['year']==year].longitude)
    x_ani, y_ani = m2(long_ani, lat_ani)
    m2.scatter(x_ani, y_ani, s=[i for i in data[data['year']==year].casualties], c= 'r')
    m2.drawcoastlines()
    m2.drawcountries()
    
ani = animation.FuncAnimation(fig,animate,list(data.year.unique()), interval = 1500)

ani.save('animation.gif', writer='imagemagick', fps=1)
plt.close(1)
filename = 'animation.gif'
video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif; base64, {0}" type="gif"  />'''.format(encoded.decode('ascii')))

most_used_word = data['motive'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(most_used_word)
word_dist = nltk.FreqDist(words)
stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(i for i in words if i not in stopwords)
wc = WordCloud(stopwords=STOPWORDS, background_color='black').generate(" ".join(words_except_stop_dist))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
