import numpy as np 

import pandas as pd 

import datetime



import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates



sns.set(style="white", color_codes=True)





dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')





FILE="../input/crime.csv"



d = pd.read_csv(FILE,

  header=0,names=['Dc_Dist', 'Psa', 'Dispatch_Date_Time', 'Dispatch_Date',

       'Dispatch_Time', 'Hour', 'Dc_Key', 'Location_Block', 'UCR_General',

       'Text_General_Code',  'Police_Districts', 'Month', 'Lon',

       'Lat'],dtype={'Dc_Dist':str,'Psa':str,

                'Dispatch_Date_Time':str,'Dispatch_Date':str,'Dispatch_Time':str,

                  'Hour':str,'Dc_Key':str,'Location_Block':str,

                     'UCR_General':str,'Text_General_Code':str,

              'Police_Districts':str,'Month':str,'Lon':str,'Lat':str},

             parse_dates=['Dispatch_Date_Time'],date_parser=dateparse)



# Fix Month to datetime Month

d['Month'] = d['Month'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m'))
d.head()
# Count of 1 for each record. Nice to have a standard column name

d['Value'] = 1





# Use sum. We're summing up all  Dc_Dist

g=d.groupby(['Month','Text_General_Code'])['Value'].sum().reset_index()



# Take a look at 'Burglary Residential'

gg=g[(g['Text_General_Code'] == 'Burglary Residential')]

gg.head()
# Create a quick plot

fig, ax = plt.subplots()

ax.plot_date(gg['Month'], gg['Value'])

ax.set_title("Burglary Residential")

fig.autofmt_xdate()

plt.show()
# Line

fig, ax = plt.subplots()

ax.plot_date(gg['Month'], gg['Value'],'k')

ax.set_title("Burglary Residential")

fig.autofmt_xdate()

plt.show()
# Red dot with Line

fig, ax = plt.subplots()

ax.plot_date(gg['Month'], gg['Value'],'k')

ax.plot_date(gg['Month'], gg['Value'],'ro')

ax.set_title("Burglary Residential")

fig.autofmt_xdate()

plt.show()
# Let's redo the grouping. Is one Dc_Dist influcing the result?

# Won't finish the analysis here...this is the a Dataset Walk-through



# Count of 1 for each record. Nice to have a standard column name

d['Value'] = 1





# Use sum. We're summing up all  Dc_Dist

g=d.groupby(['Month','Text_General_Code','Dc_Dist'])['Value'].sum().reset_index()

gg=g[(g['Text_General_Code'] == 'Burglary Residential')]

gg.head(16)
# Just picking two that might be interesting

g15=gg[gg['Dc_Dist']=='15']

g19=gg[gg['Dc_Dist']=='19']

# Red dot with Line

fig, ax = plt.subplots()



# g15

ax.plot_date(g15['Month'], g15['Value'],'k')

ax.plot_date(g15['Month'], g15['Value'],'ro')



# g19

ax.plot_date(g19['Month'], g19['Value'],'g')

ax.plot_date(g19['Month'], g19['Value'],'bo')





ax.set_title("Burglary Residential\n Dist 15 and Dist 19")

fig.autofmt_xdate()

plt.show()



# Wow these are awful colors...
# A pivot table might be interesting

pd.pivot_table(gg, values='Value', index=['Month'], columns=['Dc_Dist'], aggfunc=np.sum)