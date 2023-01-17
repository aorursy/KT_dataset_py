%matplotlib inline



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import pylab



pylab.rcParams['figure.figsize'] = (10.0, 6.0)



filename = '../input/train.csv'

data = pd.read_csv(filename, parse_dates=['Dates'], index_col='Dates')

crimes_rating = data['Category'].value_counts()

y_pos = np.arange(len(crimes_rating[0:18].keys()))



plt.barh(y_pos, crimes_rating[0:18].get_values(),  align='center', alpha=0.4, color = 'black')



plt.yticks(y_pos, map(lambda x:x.title(),crimes_rating[0:18].keys()), fontsize = 14)

plt.xlabel('Number of occurences', fontsize = 14)

plt.title('San Franciso Crimes', fontsize = 28)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))



print ('San Francisco Crimes\n')

print ('Category\t\tNumber of occurences') 

print (crimes_rating)
data['DayOfWeek'] = data.index.dayofweek

data['Hour'] = data.index.hour

data['Month'] = data.index.month

data['Year'] = data.index.year

data['DayOfMonth'] = data.index.day
import pylab

import numpy as np



pylab.rcParams['figure.figsize'] = (14.0, 8.0)



larceny = data[data['Category'] == "LARCENY/THEFT"]

assault = data[data['Category'] == "ASSAULT"]

drug = data[data['Category'] == "DRUG/NARCOTIC"]

vehicle = data[data['Category'] == "VEHICLE THEFT"]

vandalism = data[data['Category'] == "VANDALISM"]

burglary = data[data['Category'] == "BURGLARY"]



with plt.style.context('fivethirtyeight'):

    ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)

    ax1.plot(data.groupby('Hour').size(), 'ro-')

    ax1.set_title ('All crimes')

    start, end = ax1.get_xlim()

    ax1.xaxis.set_ticks(np.arange(start, end, 1))

    

    ax2 = plt.subplot2grid((3,3), (1, 0))

    ax2.plot(larceny.groupby('Hour').size(), 'o-')

    ax2.set_title ('Larceny/Theft')

    

    ax3 = plt.subplot2grid((3,3), (1, 1))

    ax3.plot(assault.groupby('Hour').size(), 'o-')

    ax3.set_title ('Assault')

    

    ax4 = plt.subplot2grid((3,3), (1, 2))

    ax4.plot(drug.groupby('Hour').size(), 'o-')

    ax4.set_title ('Drug/Narcotic')

    

    ax5 = plt.subplot2grid((3,3), (2, 0))

    ax5.plot(vehicle.groupby('Hour').size(), 'o-')

    ax5.set_title ('Vehicle')

    

    ax6 = plt.subplot2grid((3,3), (2, 1))

    ax6.plot(vandalism.groupby('Hour').size(), 'o-')

    ax6.set_title ('Vandalism')

    

    ax7 = plt.subplot2grid((3,3), (2, 2))

    ax7.plot(burglary.groupby('Hour').size(), 'o-')

    ax7.set_title ('Burglary')

  

    pylab.gcf().text(0.5, 1.03, 

                    'San Franciso Crime Occurence by Hour',

                     horizontalalignment='center',

                     verticalalignment='top', 

                     fontsize = 28)

    

plt.tight_layout(2)

plt.show()
pylab.rcParams['figure.figsize'] = (16.0, 12.0)



plt.style.use('ggplot')



daysOfWeekIdx = data.groupby('DayOfWeek').size().keys()

daysOfWeekLit = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

occursByWeek = data.groupby('DayOfWeek').size().get_values()



# Linear plot for all crimes

ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)

ax1.plot(daysOfWeekIdx, occursByWeek, 'ro-', linewidth=2)

ax1.set_xticklabels(daysOfWeekLit)

ax1.set_title ('All Crimes', fontsize=20)

# ensure that ticks are only at the bottom and left parts of the plot

ax1.get_xaxis().tick_bottom()

ax1.get_yaxis().tick_left()



# Bar plot

y = np.empty([6,7])

h = [None]*6

width = 0.1



ax2 = plt.subplot2grid((3,3), (1,0), colspan=3)



y[0] = larceny.groupby('DayOfWeek').size().get_values()

y[1] = assault.groupby('DayOfWeek').size().get_values()

y[2] = drug.groupby('DayOfWeek').size().get_values()

y[3] = vehicle.groupby('DayOfWeek').size().get_values()

y[4] = vandalism.groupby('DayOfWeek').size().get_values()

y[5] = burglary.groupby('DayOfWeek').size().get_values()



color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c','#d62728', '#9467bd', '#8c564b']



for i in range(0,6):

    h[i] = ax2.bar(daysOfWeekIdx + i*width, y[i], width, color=color_sequence[i], alpha = 0.7)



ax2.set_xticks(daysOfWeekIdx + 3*width)

ax2.set_xticklabels(daysOfWeekLit)

# ensure that ticks are only at the bottom and left parts of the plot

ax2.get_xaxis().tick_bottom()

ax2.get_yaxis().tick_left()



ax2.legend((item[0] for item in h), 

           ('Larceny', 'Assault', 'Drug', 'Vehicle', 'Vandalism', 'Burglary'), 

           bbox_to_anchor=(0.88, 1), loc=2, borderaxespad=0., frameon=False)



pylab.gcf().text(0.5, 1.00, 

            'San Franciso Crime Occurence by Day Of Week',

            horizontalalignment='center',

            verticalalignment='top', 

             fontsize = 28)



plt.show()
pylab.rcParams['figure.figsize'] = (16.0, 8.0)



monthsIdx = data.groupby('Month').size().keys() - 1

monthsLit = ['January', 'February', 

             'March', 'April', 'May', 

             'June', 'July','August', 

             'September', 'October', 'Novemeber', 'December']

occursByMonth = data.groupby('Month').size().get_values()



# Linear plot for all crimes

ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)

ax1.plot(monthsIdx, occursByMonth, 'ro-', linewidth=2)



ax1.set_title ('All Crimes', fontsize=20)



start, end = ax1.get_xlim()

ax1.xaxis.set_ticks(np.arange(start, end, 1))

ax1.set_xticklabels(monthsLit)

# ensure that ticks are only at the bottom and left parts of the plot

ax1.get_xaxis().tick_bottom()

ax1.get_yaxis().tick_left()



# Linear normalized plot for 6 top crimes

ax2 = plt.subplot2grid((3,3), (1,0), colspan=3, rowspan=2)



y = np.empty([6,12])

y[0] = larceny.groupby('Month').size().get_values()

y[1] = assault.groupby('Month').size().get_values()

y[2] = drug.groupby('Month').size().get_values()

y[3] = vehicle.groupby('Month').size().get_values()

y[4] = vandalism.groupby('Month').size().get_values()

y[5] = burglary.groupby('Month').size().get_values()



crimes = ['Larceny/theft', 'Assault', 'Drug', 'Vehicle', 'Vandalism', 'Burglary']

color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c','#d62728', '#9467bd', '#8c564b']



for i in range(0,6):

    y[i]= (y[i]-min(y[i]))/(max(y[i])-min(y[i]))  # normalization

    h[i] = ax2.plot(monthsIdx, y[i],'o-', color=color_sequence[i], lw=2)



ax2.set_ylabel("Crime occurences by month, normalized")



ax2.xaxis.set_ticks(np.arange(start, end+2, 1))

ax2.set_xticklabels(monthsLit)



ax2.legend((item[0] for item in h), 

           crimes, 

           bbox_to_anchor=(0.87, 1), loc=2, borderaxespad=0., frameon=False)



pylab.gcf().text(0.5, 1.00, 

            'San Franciso Crime Occurence by Month',

            horizontalalignment='center',

            verticalalignment='top', 

             fontsize = 28)

plt.show()
pylab.rcParams['figure.figsize'] = (16.0, 10.0)



years = data.groupby('Year').size().keys()

occursByYear = data.groupby('Year').size().get_values()



# Linear plot for all crimes

ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)

ax1.plot(years, occursByYear, 'ro-', linewidth=2)



ax1.set_title ('All Crimes', fontsize=20)



start, end = ax1.get_xlim()

ax1.xaxis.set_ticks(np.arange(start, end, 1))

# ensure that ticks are only at the bottom and left parts of the plot

ax1.get_xaxis().tick_bottom()

ax1.get_yaxis().tick_left()



# Linear normalized plot for 6 top crimes

ax2 = plt.subplot2grid((3,3), (1,0), colspan=3, rowspan=2)



y = np.empty([6,13])

y[0] = larceny.groupby('Year').size().get_values()

y[1] = assault.groupby('Year').size().get_values()

y[2] = drug.groupby('Year').size().get_values()

y[3] = vehicle.groupby('Year').size().get_values()

y[4] = vandalism.groupby('Year').size().get_values()

y[5] = burglary.groupby('Year').size().get_values()



for i in range(0,6):

    h[i] = ax2.plot(years, y[i],'o-', color=color_sequence[i], lw=2)



ax2.set_ylabel("Crime occurences by year")



start, end = ax2.get_xlim()  

ax2.xaxis.set_ticks(np.arange(start, end+2, 1))



ax2.legend((item[0] for item in h), 

           crimes, 

           bbox_to_anchor=(0.87, 1), loc=2, borderaxespad=0., frameon=False)



pylab.gcf().text(0.5, 1.00, 

            'San Franciso Crime Occurence by Year',

            horizontalalignment='center',

            verticalalignment='top', 

             fontsize = 28)

plt.show()
pylab.rcParams['figure.figsize'] = (16.0, 5.0)

yearMonth = data.groupby(['Year','Month']).size()

ax = yearMonth.plot(lw=2)

plt.title('San Franciso Crimes Trend by Month&Year', fontsize=24)

plt.show()