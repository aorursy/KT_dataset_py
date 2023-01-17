import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

from mpl_toolkits.basemap import Basemap



import plotly.plotly as py

import plotly.graph_objs as go



import seaborn as sns

sns.set_style('whitegrid')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
t_data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1',

                    usecols=[0, 1, 2, 3, 7, 9, 11, 12, 13, 14, 15, 19, 20, 21, 26, 28, 29, 34, 35, 36, 58, 68, 69, 81, 98, 101, 104, 134]) 



# Fill empty cells with proper value

t_data['nkill'] = t_data['nkill'].fillna(0).astype(int)

t_data['nwound'] = t_data['nwound'].fillna(0).astype(int)



t_data.head()
plt.figure(figsize=(9,7))

ax = sns.countplot(y="attacktype1_txt", data=t_data)

ax.set_xlabel("Amount of type")

ax.set_ylabel("Attack type")
orange_palette = ((3, 0, '#FBBC00', '1 - 20'), (4, 20, '#FDA600', '21 - 50'), (5, 50, '#EE8904', '51 - 100'), \

                  (7, 100, '#ED9001', '101 - 250'), (9, 250, '#ED6210', '251 - 600'), \

                  (11, 600, '#DE6D0A', '601 - 1000'), (13, 1000, '#D8510F', '1001 - 2000'), \

                  (15, 2000, '#D23711', '2001 - 4000'), (18, 4000, '#F61119', '4001 - 7500'), \

                  (30, 7500, '#9C200A', '7501 - ∞')) #marker size, count size, color



plt.figure(figsize=(15,15))

# Rounds the long- and latitude to a number withouth decimals, groups them on long- and latitude and counts the amount of attacks.

df_coords = t_data.round({'longitude':0, 'latitude':0}).groupby(["longitude", "latitude"]).size().to_frame(name = 'count').reset_index()

m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

m.drawcoastlines()

m.shadedrelief()

    

def plot_points(marker_size, count_size, colour, label_count):

    x, y = m(list(df_coords.longitude[df_coords['count'] >= count_size].astype("float")),\

                (list(df_coords.latitude[df_coords['count'] >= count_size].astype("float"))))

    points = m.plot(x, y, "o", markersize = marker_size, color = colour, label = label_count, alpha = .5)



for p in orange_palette:

    plot_points(p[0], p[1], p[2], p[3]) 

    

plt.title("Amount of terrorist attacks per rounded coordinates", fontsize=24)

plt.legend(title= 'Colour per counted attack', loc ='lower left', prop= {'size':11})

plt.show()
sns.jointplot(x='longitude', y='latitude', data=df_coords, kind="hex", color="#4CB391", size=15, stat_func=None, edgecolor="#EAEAF2", linewidth=.2)

plt.title('Amount of terrorist attacks per rounded coordinates')
df_day_coords = t_data[['imonth', 'iday', 'longitude', 'latitude', 'success']].copy()[(t_data['iday'] != 0) & (t_data['imonth'] != 0)]
fig, ax = plt.subplots(figsize=(14,10))

sns.countplot(x="iday", data=df_day_coords, ax=ax, palette=sns.cubehelix_palette(15, start=.3, rot=.3))

ax.set_xlabel('Day of the month')

ax.set_ylabel('Amount of terrorist attacks')
fig, axs = plt.subplots(nrows=12)

fig.set_size_inches(15, 100, forward=True)



for i in range(1,13):

    monthly_data = df_day_coords[df_day_coords['imonth'] == i]

    sns.countplot(x="iday", data=monthly_data, hue="success", ax=axs[i-1])

    axs[i-1].set_xlabel('Day of the month')

    axs[i-1].set_ylabel('Amount of terrorist attacks')
succes_month = sns.factorplot(x="imonth", hue="success", 

                                  kind="count", data=df_day_coords, size=10, palette="muted")
#I would love to enlarge the graphs in the vertical direction however was unable to achieve this

# I tried plot2grid and subplot and regular plot however without success



def generate_graph(by_region_list):

    fig = plt.figure(figsize=(15,10))

    i = 1

    

    for element in by_region_list:

        ax1 = fig.add_subplot(11,2,i)

        ax1.set(title = '#Attacks region %s ' % region_dictionary[element[2]],

                ylabel = 'Attack count', xlabel = 'year')



        #entering data

        ax1.plot(element[0].index, element[0].eventid, label = 'Successfull attacks' )

        ax1.plot(element[1].index, element[1].eventid, label = 'Failed attacks' )

        

        i+=1

    

    #add legend

    ax1.legend(loc = 'upper center', frameon = True, edgecolor = 'black', bbox_to_anchor =(-0.1,-0.4))

    plt.show()  





def by_region():

        for region_number in region_dictionary:

            region_data = t_data[(t_data.region == region_number)] #for each region group data by year

            region_grouped_success = region_data[(region_data.success == 1)].groupby('iyear').count() #filter on success and group by year

            region_grouped_failure = region_data[(region_data.success == 0)].groupby('iyear').count() #filter on failure and group by year

            

            by_region_list.append([region_grouped_success, region_grouped_failure, region_number])

        

        #create line plot for region grouped by year

        generate_graph(by_region_list)



by_region_list = []

by_region()
def multi_graph(result,result_list, xmin, xmax, ymin, ymax):

    fig2, ax2 = plt.subplots(figsize = (15,8))

    number = 1 #the for-loop in append_list processes the regions in order from 1 to 12

    for j in result_list:

        ax2.plot(j.index, j.eventid, label = '%s ' % region_dictionary[number] )

        number += 1



    plt.xlim([xmin,xmax])

    plt.ylim([ymin,ymax])

    plt.xlabel('year')

    plt.ylabel('number of attacks')

    plt.title(result)

    ax2.legend(loc = 'center', frameon = True, edgecolor = 'black',bbox_to_anchor =(1.2,0.4))





success_list = []

failure_list = []



for i in region_dictionary:

    region_data = t_data[(t_data.region == i)]

    region_data_success = region_data[(region_data.success == 1)]

    region_data_failure = region_data[(region_data.success == 0)]

    region_grouped_success = region_data_success.groupby('iyear').count()

    region_grouped_failure = region_data_failure.groupby('iyear').count()



    

    success_list.append(region_grouped_success)

    failure_list.append(region_grouped_failure)



multi_graph('Successes',success_list, 1970, 2011, 0, 2100)

multi_graph('Successes',success_list, 2012, 2016, 0, 6500)

multi_graph('Failures',failure_list, 1970, 2011, 0, 200)

multi_graph('Failures',failure_list, 2012, 2016, 0, 1300)
# toelichting

# Wel of niet gelukt en waarom?