import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
Crv = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

Crv.head()
# Now lets convert the Date and Last updated to Datetime.

Crv['Date'] = pd.to_datetime(Crv['Date'])

Crv['Last Update'] = pd.to_datetime(Crv['Last Update'])
#Lets see how our data loooks now

Crv.head()
# Let us arrange the date and last update to day, week and year

Crv['Day'] = Crv['Date'].dt.day

Crv['Month'] = Crv['Date'].dt.month

Crv['Week'] = Crv['Date'].dt.week

Crv['WeekDay'] = Crv['Date'].dt.weekday
#Let us see our data Once again

Crv.head()
# Lets bring out the needed columns for our analysis as Ned

Ned = ['Confirmed','Deaths','Recovered']
import matplotlib.gridspec as gridspec
def multi_plot():

    fig = plt.figure(constrained_layout=True, figsize=(15,8))

    grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)



    ax1 = fig.add_subplot(grid[0, :2])

    ax1.set_title('Daily Reports')

    Crv.groupby(['Date']).sum()[Ned].plot(ax=ax1)



    ax2 = fig.add_subplot(grid[1, :2])

    ax2.set_title('Monthly Reports')

    Crv.groupby(['Month']).sum()[Ned].plot(kind='bar',ax=ax2)

    ax2.set_xticklabels(range(1,3))



    ax3= fig.add_subplot(grid[0, 2:])

    ax3.set_title('Weekly Reports')

    weekdays = Crv.groupby('Week').nth(-1)['Date']

    Crv[Crv['Date'].isin(weekdays)].groupby('Date')[Ned].sum().plot(kind='bar',ax=ax3)

    ax3.set_xticklabels(range(1,len(weekdays)+1))



    ax4 = fig.add_subplot(grid[1, 2:])

    ax4.set_title('WeekDays Reports')

    Crv.groupby(['WeekDay']).sum()[Ned].plot(ax=ax4)

    plt.tight_layout()

multi_plot()
# From the count and chart the latest case count = today case count + previous days case count. 

# We will take the latest date as the updated count.
#LD as Last Date

#LU as Last Update

LD = Crv['Date'].iloc[-1]

LU = Crv[Crv['Date'].dt.date == LD]
LU.head()
Crv['Country'].value_counts()
# Mainland china and China are the same so, we combine both figures.

Crv['Country'].replace({'Mainland China':'China'},inplace=True)
# Lets check the value count for unique countries with the recent date case count

Crv[Crv['Date'] != Crv['Last Update']]['Country'].value_counts()
confirmedCase = int(LU['Confirmed'].sum())

deathCase = int(LU['Deaths'].sum())

recoveredCase = int(LU['Recovered'].sum())

print("No of Confirmed cases globally {}".format(confirmedCase))

print("No of Recovered case globally {}".format(recoveredCase))

print("No of Death case globally {}".format(deathCase))
TC10 =  LU.groupby(['Country']).sum().nlargest(10,['Confirmed'])[Ned]

TC10

print("Top 10 Countries that are affected")

print(TC10)
plt.figure(figsize=(12,6))

plt.xticks(rotation=90)

plt.title("Top 10 Countries that are affected by Coronavirus")

sns.barplot(x=TC10.index,y='Confirmed',data=TC10,color ='red')

plt.show()
def xtra_plot():

    fig = plt.figure(constrained_layout=True, figsize=(15,8))

    grid = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)



    ax1 = fig.add_subplot(grid[0, :2])

    ax1.set_title('Daily Reports')

    LU.groupby(['Date']).sum()[Ned].plot(ax=ax1)



    ax2 = fig.add_subplot(grid[1, :2])

    ax2.set_title('Monthly Reports')

    LU.groupby(['Month']).sum()[Ned].plot(kind='bar',ax=ax2)

    ax2.set_xticklabels(range(1,3))



    ax3= fig.add_subplot(grid[0, 2:])

    ax3.set_title('Weekly Reports')

    weekdays = LU.groupby('Week').nth(-1)['Date']

    LU[LU['Date'].isin(weekdays)].groupby('Date')[Ned].sum().plot(kind='bar',ax=ax3)

    ax3.set_xticklabels(range(1,len(weekdays)+1))



    ax4 = fig.add_subplot(grid[1, 2:])

    ax4.set_title('WeekDays Reports')

    LU.groupby(['WeekDay']).sum()[Ned].plot(ax=ax4)

    plt.tight_layout()

multi_plot()