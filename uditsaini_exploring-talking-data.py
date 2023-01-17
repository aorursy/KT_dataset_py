%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import seaborn as sns
sns.set(color_codes=True)
app_event=pd.read_csv("../input/events.csv")
app_event.shape
app_event.timestamp=pd.to_datetime(app_event.timestamp)
app_event['time_hour'] = app_event.timestamp.apply(lambda x: x.hour)
app_event['time_hour'].value_counts()
#event frequency by hour

ax = sns.countplot(x="time_hour", data=app_event)
import calendar
app_event['week_day'] = app_event.timestamp.apply(lambda x: calendar.day_name[x.weekday()])
ax = sns.countplot(x="week_day", data=app_event)
gender=pd.read_csv("../input/gender_age_train.csv")
print(gender.gender.value_counts())
ax = sns.countplot(x="gender", data=gender)
sns.distplot(gender.age, hist=False);
#here age is not normally distributaed between 20 - 40 are the dominate age 
print("Age distribution for male and female Female at old age are using more mobile devises then males")
sns.kdeplot(gender.age[gender.gender=="M"], label="Male")
sns.kdeplot(gender.age[gender.gender=="F"],  label="Female")
plt.legend();
print("Male age droup count")
ax = sns.countplot(x="group", data=gender[gender.gender=="M"])
print("Female age droup count")
ax = sns.countplot(x="group", data=gender[gender.gender=="F"])
gamescat=pd.read_csv("../input/label_categories.csv")
print(gamescat.head())
print(gamescat.shape)
import sys
pd.options.display.encoding = "UTF-8"
phone_brand=pd.read_csv("../input/phone_brand_device_model.csv",encoding="UTF-8")
print(phone_brand.shape)
print(phone_brand.head())
#this script has been taken from 
#https://www.kaggle.com/ramirogomez/talkingdata-mobile-user-demographics/talking-data-overview
fig = plt.figure(figsize=(16, 12))
markersize = 1
markertype = ',' # pixel
markercolor = '#444444'
markeralpha = .8 #  a bit of transparency
# http://isithackday.com/geoplanet-explorer/index.php?woeid=23424781
# Location (lat/lon): 36.894402, 104.166
# Bounding Box:
# NE 53.5606, 134.773605
# SW 15.77539, 73.557701
m = Basemap(
    projection='mill', lon_0=104.166, lat_0=36.894402,
    llcrnrlon=73.557701, llcrnrlat=15.77539,
    urcrnrlon=134.773605, urcrnrlat=53.5606)

# avoid border around map
m.drawmapboundary(fill_color='#ffffff', linewidth=.0)

# draw event locations
x, y = m(app_event.longitude.values, app_event.latitude.values)
m.scatter(x, y, markersize, marker=markertype, color=markercolor, alpha=markeralpha)
# annotations
plt.annotate('Talking Data Event Locations', xy=(0.02, .96), size=20, xycoords='axes fraction')
footer = 'Author: Ramiro Gomez - ramiro.org | Data: TalkingData provided by kaggle.com'
plt.annotate(footer, xy=(0.02, -0.02), size=14, xycoords='axes fraction')

plt.show()