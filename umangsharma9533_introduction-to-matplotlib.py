# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
seattle_weather=pd.read_csv('/kaggle/input/seattle_weather (1).csv')

austin_weather=pd.read_csv('/kaggle/input/austin_weather.csv')
plt.figure(figsize=(50,50))

# Create a Figure and an Axes with plt.subplots

fig, ax = plt.subplots()

ax.plot(seattle_weather["DATE"], seattle_weather["MLY-PRCP-NORMAL"],label='Seattle',marker="o",linestyle="None")

ax.plot(austin_weather["DATE"], austin_weather["MLY-PRCP-NORMAL"],label='Austin',marker="v")



# Customize the x-axis label

ax.set_xlabel("Time (months)")



# Customize the y-axis label

ax.set_ylabel("Precipitation (inches)")



# Add the title

ax.set_title("Weather patterns in Austin and Seattle")

ax.legend()

# Display the figure

plt.show()
plt.figure(figsize=(50,50))

# Create a Figure and an Axes with plt.subplots

fig, ax = plt.subplots()

ax.hist(seattle_weather["MLY-PRCP-NORMAL"],label='Seattle',bins=20)

ax.hist(austin_weather["MLY-PRCP-NORMAL"],label='Austin',bins=20)



# Customize the x-axis label

ax.set_xlabel("Precipitation Total in (inches)")



# Customize the y-axis label

ax.set_ylabel("Number of times in year")



# Add the title

ax.set_title("Weather patterns in Austin and Seattle")

ax.legend()

# Display the figure

plt.show()
plt.figure(figsize=(100,50))

# Create a Figure and an Axes with plt.subplots

fig, ax = plt.subplots(2,1)

ax[0].hist(seattle_weather["MLY-PRCP-NORMAL"],label='Seattle',bins=20)

ax[1].hist(austin_weather["MLY-PRCP-NORMAL"],label='Austin',bins=20,color="r")



# Customize the x-axis label

ax[1].set_xlabel("Precipitation Total in (inches)")



# Customize the y-axis label

ax[0].set_ylabel("Frequency over year")

ax[1].set_ylabel("Frequency over year")



# Add the title

ax[0].set_title("Weather patterns in Austin and Seattle")

ax[0].legend()

ax[1].legend()

# Display the figure

plt.show()
climate_change=pd.read_csv('/kaggle/input/climate_change.csv',parse_dates=True, index_col='date')
fig,ax=plt.subplots()

plt.figure(figsize=(60,60))

ax.plot(climate_change.index,climate_change['relative_temp'])

ax.set_xlabel("Time")

ax.set_ylabel("Relative Temp over years")

plt.show()
fig,ax=plt.subplots()

plt.figure(figsize=(60,60))

TwentyFirstCentuary=climate_change['2010-01-01':'2020-12-31']

ax.plot(TwentyFirstCentuary.index,TwentyFirstCentuary['relative_temp'])

ax.set_xlabel("Time(years in decade)")

ax.set_ylabel("Relative Temp over years")

plt.show()
fig,ax=plt.subplots()

plt.figure(figsize=(60,60))

TwentySixteen=climate_change['2016-01-01':'2016-12-31']

ax.plot(TwentySixteen.index,TwentySixteen['relative_temp'],marker="v",color='r')

ax.set_xlabel("Time(months)")

ax.set_ylabel("Relative Temp over years")

plt.show()
fig,ax=plt.subplots()

plt.figure(figsize=(60,60))

ax.plot(climate_change.index,climate_change['relative_temp'],color='b')

ax2=ax.twinx()

ax2.plot(climate_change.index,climate_change['co2'],color='r')

ax.set_xlabel("Time")

ax.set_ylabel("Relative Temp over years",color='b')

ax.tick_params('y',colors='b')

ax2.set_ylabel("Co2 over years",color='r')

ax2.tick_params('y',colors='r')

plt.show()
def plotTimeSeriesData(axes,x,y,xlabel,ylabel,color_name):

    axes.plot(x,y,color=color_name)

    axes.set_xlabel(xlabel)

    axes.set_ylabel(ylabel,color=color_name)

    axes.tick_params('y',colors=color_name)    
fig,ax=plt.subplots()

plt.figure(figsize=(60,60))

plotTimeSeriesData(ax,climate_change.index,climate_change['relative_temp'],'Time',"Relative Temp over years",'b')

ax2=ax.twinx()

plotTimeSeriesData(ax2,climate_change.index,climate_change['co2'],'Time',"co2 over years",'r')
fig,ax=plt.subplots()

plt.figure(figsize=(60,60))

plotTimeSeriesData(ax,climate_change.index,climate_change['relative_temp'],'Time',"Relative Temp over years",'b')

ax2=ax.twinx()

plotTimeSeriesData(ax2,climate_change.index,climate_change['co2'],'Time',"co2 over years",'r')

ax.annotate(">1 degree", xytext=(pd.Timestamp('2008-10-06'),-0.2), xy=(pd.Timestamp('2015-10-06'),1), arrowprops={"arrowstyle":"->","color":"gray"})

plt.show()
# Index_Col =0 means treat first column in the csv file as index

medals=pd.read_csv('/kaggle/input/medals_by_country_2016.csv',index_col=0)
fig,ax=plt.subplots()

plt.figure(figsize=(50,50))

ax.bar(medals.index,medals["Gold"])

ax.set_xlabel('Countries wise')

ax.set_ylabel('Number of Gold medals')

plt.show()
# ax.set_xticklabel(label vlaues,rotation=in degrees), used to rotate the labels by user mentioned angle

fig,ax=plt.subplots()

plt.figure(figsize=(50,50))

ax.bar(medals.index,medals["Gold"])

ax.set_xticklabels(medals.index,rotation=90)

ax.set_xlabel('Countries wise')

ax.set_ylabel('Number of Gold medals')

plt.show()
fig,ax=plt.subplots()

plt.figure(figsize=(50,50))

ax.bar(medals.index,medals["Gold"],label="Gold")

ax.bar(medals.index,medals["Silver"],bottom=medals['Gold'],label='Silver')

ax.bar(medals.index,medals["Bronze"], bottom=medals['Gold']+medals['Silver'], label="Bronze")

ax.set_xticklabels(medals.index,rotation=90)

ax.set_xlabel('Countries wise')

ax.set_ylabel('Number of Gold medals')

ax.legend()

plt.show()
mens_rowing=pd.read_csv('/kaggle/input/summer2016.csv')

fig,ax=plt.subplots()

plt.figure(figsize=(50,50))

ax.bar("Rowing",mens_rowing["Height"].mean(),yerr=mens_rowing["Height"].std())

plt.show()
plt.style.use("default")

fig, ax = plt.subplots()



# Add Seattle temperature data in each month with error bars

ax.errorbar(seattle_weather["DATE"],seattle_weather["MLY-TAVG-NORMAL"],yerr=seattle_weather["MLY-TAVG-STDDEV"])

ax.errorbar(austin_weather["DATE"],austin_weather["MLY-TAVG-NORMAL"], yerr=austin_weather["MLY-TAVG-STDDEV"])

ax.set_ylabel("Temperature (Fahrenheit)")



plt.show()

fig,ax=plt.subplots()

ax.hist(mens_rowing['Height'])



ax.set_xlabel('Height')

ax.set_ylabel('Frequency')

ax.legend()

plt.show()
fig,ax=plt.subplots()

ax.hist(mens_rowing['Height'],histtype="step")#Histtype=step will create a graph of hollow lines



ax.set_xlabel('Height')

ax.set_ylabel('Frequency')

ax.legend()

plt.show()
mens_gymnastics=mens_rowing[mens_rowing["Sport"]=="Gymnastics"]
mens_rowings=mens_rowing[mens_rowing["Sport"]=="Rowing"]
fig,ax=plt.subplots()

ax.boxplot([mens_rowing['Height'],mens_gymnastics["Height"]])

ax.set_xticklabels(["Rowing","Gymnastics"])

ax.set_ylabel("Height(cm)")

plt.show()
fig,ax=plt.subplots()

ax.scatter(climate_change["co2"],climate_change["relative_temp"])

ax.set_xlabel("CO2(ppm)")

ax.set_ylabel("Relative temperature (Celsius)")

plt.show()
eighties=climate_change["1980-01-01": "1989-12-31"]

nineties=climate_change["1990-01-01": "1999-12-31"]

fig,ax=plt.subplots()

ax.scatter(eighties['co2'],eighties['relative_temp'],color="red",label="eighties")

ax.scatter(nineties['co2'],nineties['relative_temp'],color="blue",label="nineties")

ax.legend()

ax.set_xlabel("Co2(ppm)")

ax.set_ylabel("Relative temperature (Celsius)")

plt.show()
# Plot style

# We can change style of the plot by using plt.style.use("Name of the style")
plt.style.use("bmh")

eighties=climate_change["1980-01-01": "1989-12-31"]

nineties=climate_change["1990-01-01": "1999-12-31"]

fig,ax=plt.subplots()

ax.scatter(eighties['co2'],eighties['relative_temp'],color="red",label="eighties")

ax.scatter(nineties['co2'],nineties['relative_temp'],color="blue",label="nineties")

ax.legend()

ax.set_xlabel("Co2(ppm)")

ax.set_ylabel("Relative temperature (Celsius)")

plt.show()
# TO save figure we can use fig.savefig('Name required')
plt.style.use("bmh")

eighties=climate_change["1980-01-01": "1989-12-31"]

nineties=climate_change["1990-01-01": "1999-12-31"]

fig,ax=plt.subplots()

ax.scatter(eighties['co2'],eighties['relative_temp'],color="red",label="eighties")

ax.scatter(nineties['co2'],nineties['relative_temp'],color="blue",label="nineties")

ax.legend()

ax.set_xlabel("Co2(ppm)")

ax.set_ylabel("Relative temperature (Celsius)")

plt.show()

fig.savefig("co.png")