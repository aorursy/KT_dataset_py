#Dataset - https://www.kaggle.com/lampubhutia/nyc-flight-delay



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport #Pandas profiling to understand the data if required 

import datetime #datettme library

import math #math library



import seaborn as sns

from matplotlib import pyplot as plt





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read flights dataset csv file available on Kaggle

df=pd.read_csv("../input/nyc-flight-delay/flight_data.csv")

dt=df

df.head(5) # Read the top 5 rows of the dataset 



#import calendar

#df.month = df.month.apply(lambda x: calendar.month_abbr[x])   # i didnt convert the month into Jan , Feb month format to keep it simple analysis as of now 
#call shape to identify the rows and columns of the dataset

print('Flights dataset is having ',df.shape[0] , 'rows and ',df.shape[1],' columns which is big enough for an excel file to handle/process sometimes')
#Simple function to check Null values in each column in the dataset

def check_NullValues():

    out=df.isnull().sum()

    found=0

    for counter in out.index:

        if out[counter]>0:

            found=found+1

            print( "column", counter , " is having ",out[counter] , "Null values")





    if(found==0):

        print("No Null values found in dataframe")
#Show Null value columns alongwith quantity of Null values 

check_NullValues()
#Drop Null Values

df.dropna(axis=0,how ='any', inplace=True)



#Drop the columns that i dont need for my analysis to reduce the processing time/power

df.drop(axis=1,columns=['dep_time','sched_dep_time','sched_arr_time','arr_time','time_hour'],inplace=True)
#Let's call null value check function again to check if we still have any null values. i know it shouldnt be but playing by calling function again to test it 

check_NullValues()
#call shape to identify the rows and columns of the dataset

print('Flights dataset is having ',df.shape[0] , 'rows now and ',df.shape[1],' columns as we have dropped 5 columns also ')

print('this dataset is simple dataset which doesnt need much of cleaning/prep unlike real world dataset where we need to spend so much of time in data preparations')
'''

Function draw_barplot is a function to draw bar plots for visual analysis 



#_style = variable for seaborn style

#_x = X Axis variable

#_y = Y Axis variable

#_dataset= dataset variable

#_suptitle = Subtitle Text to be displayed

#_xLabels = Labels to be displayed for x axis 

#_yLabels = Labels to be displayed for y axis 



'''

        

def draw_barplot(_style, _x,_y,_dataset,_suptitle,_xLabels,_yLabels):        

            sns.set(style=_style)         

            g = sns.catplot(x=_x, y=_y, data=_dataset,aspect=2,

                            height=5, kind="bar", palette="muted")

            plt.subplots_adjust(top=0.9)

            g.fig.suptitle(_suptitle)

            g.set_xlabels(_xLabels)

            g.set_ylabels(_yLabels)    
'''

Function plot_CountPlot is a function to draw Count plots for visual analysis 



'''



def plot_CountPlot(X,Dataframe,Title):

    fig = plt.figure(figsize=(15,5))

    ax = sns.countplot(x=X, data=Dataframe ,palette='pastel' ,edgecolor=sns.color_palette("dark", 3))

    ax.set_title(Title)

    ax.legend(loc='upper right')



    for t in ax.patches:

        if (np.isnan(float(t.get_height()))):

            ax.annotate(0, (t.get_x(), 0))

        else:

            ax.annotate(str(format(int(t.get_height()), ',d')), (t.get_x(), t.get_height()*1.01))

    plt.show();   

    
dt=df.groupby(by='origin').count().reset_index().sort_values(by='year', ascending=False)

dt['flightsCount']=dt.year



#Call Function to draw Bar Graph

draw_barplot(_style='whitegrid',_x='origin',_y='flightsCount',_xLabels='Airport',

             _yLabels='Number of Flights', _dataset=dt,

             _suptitle="Number of Flights from different Airports"

            )

plot_CountPlot('origin',df,'Number of Flights from different Airports')


delay_Frame=df.groupby(by='origin').mean().reset_index().sort_values(by='year', ascending=False)

dt[['origin','dep_delay']]



#Call Function to draw Bar Graph

draw_barplot(_style='whitegrid',_x='origin',_y='dep_delay',_xLabels='Airport',

             _yLabels='Average Departure Delay', _dataset=delay_Frame,

             _suptitle="Average Departure Delay from Airport"

            )

delay_Frame=df.groupby(by='origin').mean().reset_index().sort_values(by='year', ascending=False)

dt[['origin','arr_delay']]



#Call Function to draw Bar Graph

draw_barplot(_style='whitegrid',_x='origin',_y='arr_delay',_xLabels='Airport',

             _yLabels='Average Arrival Delay', _dataset=delay_Frame,

             _suptitle="Average Arrival Delay at Airport"

            )

plot_CountPlot('carrier',df,'Flights Carrier Frequency')
##Let's calculate the flight speed first as we dont have flight speed in the dataset 

df['aircraft_speed']=(np.floor(df.distance/df.air_time)*60).astype(int)

df=df.sort_values("aircraft_speed", axis = 0, ascending = False)


sns.set_style('whitegrid')

plt.figure(figsize=(30,10))



#Below visualisations tell us about the speed variations of the Aircrafts 



# Violin plot

ax=sns.violinplot(x='carrier', y='aircraft_speed', 

               data=df,palette='muted')



plt.title('Aircraft Speed Analysis based on Carrier')

ax.set(xlabel='Aircarft Carrier', ylabel='Aircraft Speed')









#Box and Whisker Graph

plt.figure(figsize=(20,10))

ax=sns.boxplot(x="carrier", y="aircraft_speed",             

            data=df)

sns.despine(offset=10, trim=True)

plt.title('Aircraft Speed Analysis based on Carrier')

#ax.set_xlabels("Airport")

#ax.set_ylabels("Average Departure Delay")

ax.set(xlabel='Aircarft Carrier', ylabel='Aircraft Speed')
destflightcountdf=df.groupby(["dest"],sort=True).count()

destflightcountdf['FlightsCount']=destflightcountdf['dep_delay']

destflightcountdf=destflightcountdf['FlightsCount'].sort_values(ascending=False).head(5)

destflightcountdf

# Pie chart

labels = destflightcountdf.index

sizes = destflightcountdf.values

# only "explode" the 2nd slice (i.e. 'Hogs')

explode = (0.1, 0, 0, 0,0)

#add colors

colors = ['#ff9999','#66b3ff','#99ff99','#ffbb99','#00bb99']

fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, colors=colors, 

        shadow=True, startangle=135)

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')

plt.title('Top 5 - Maximum number of flights headed towards below Airport')

plt.tight_layout()

plt.show()

ota=(df[df["arr_delay"]==0].groupby("origin").count()).loc[:,"year":"month"]

OnTimeArrivalPerc=np.round(((ota["year"]/ df["month"].count() ) * 100),decimals=2)



plt.plot(OnTimeArrivalPerc)

plt.title("On Time Arrival % Analysis")

plt.ylabel("On Time Arrival %")

plt.xlabel("Origin Airports")