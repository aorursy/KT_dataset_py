# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')
# TO-DO: Removing the unnecessary columns, and combine the row by the country



# Let's see what is in the first csv

df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')



print('Before:\n')

print(df.info(), '\n')



# Remove the unnecessary columns

df = df.drop(['SNo','Last Update'], axis=1)



# data columns for the anlysis

cols = ['Confirmed','Deaths','Recovered']

df.columns = ['Date', 'Province/State', 'Country'] + cols



# Change datetime string values of 'Date' Column to date.

df['Date'] =  list(i.date() for i in map( pd.to_datetime, df['Date']))

updated = df['Date'].max()

print(f'Data has been updated on {updated}\n')



# Aggregate the data by Contry

df = df.groupby(['Date', 'Country'], as_index=False).sum()



# Convert all the float data to integer

df[cols] = df[cols].astype(int)



# 'Mainland China' and 'China' are the names of the same country. Give the new same name 'China' to them 

df.loc[df['Country'] == 'Mainland China','Country'] = 'China'



# 'Others' stands the `Diamond Princess cruise ship` case. Medea and statistics normally combine this to Japan case.

# (https://edition.cnn.com/asia/live-news/coronavirus-outbreak-02-21-20-intl-hnk/h_476c5e1422d72d7ff94563138103b7b1)

# So let's combine this to Japan case.

is_japan = df['Country'] == 'Japan'

is_others = df['Country'] == 'Others'

df.loc[is_japan,cols] = df.loc[is_japan].set_index('Date')[cols].add(df.loc[is_others].set_index('Date')[cols],fill_value=0).values

# and drop the 'Others' data

df = df.drop(df[is_others].index)



print('After:\n')

print( df.info(), '\n\n')

df
import pandas as pd



# Read the population data

dfpop = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')

print('After:\n')

print( dfpop.info(), '\n\n')



# Rename the Columns

dfpop.columns = ['Country', 'Population', 'Yearly Change',

       'Net Change', 'Density (P/Km²)', 'Land Area (Km²)', 'Migrants (net)',

       'Fert. Rate', 'Med. Age', 'Urban Pop %', 'World Share']



# Set 'Country' as the index

dfpop = dfpop.set_index('Country')



# Drop unnecessary columns

dfpop = dfpop.drop(['Yearly Change', 'Net Change','Migrants (net)','Fert. Rate'], axis=1)



# Rename some country to match the other DataFrame

s = ['Macao','United Kingdom','United States']

t = ['Macau', 'UK','US']

as_list = dfpop.index.tolist()



for i,j in zip(s,t):

    idx = as_list.index(i)

    as_list[idx] = j 

dfpop.index = as_list



print('After:\n')

print( dfpop.info(), '\n\n')

dfpop.head()
import pandas as pd



# Get gio informatioin from another dataset

df2 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

df2 = df2.groupby('Country/Region',as_index=False).agg({'Lat':'mean','Long':'mean'})

# 'Mainland China' and 'China' are the names of the same country. Give the new same name 'China' to them 

df2.loc[df2['Country/Region']=='Mainland China','Country/Region']= 'China'

df2 = df2.set_index('Country/Region')

df2.index.name = 'Country'

# 'Others' is Diamond Princess Cruise Cases the number of which has been  included into Janpan, so just drop it. 

df2 = df2.drop(df2[df2.index == 'Others'].index)



# Add the latest case numbers to the dataset 

df2 = df2.join(df[df['Date']==updated].set_index('Country'))

# Add the country popluation information to the dataset 

df2 = df2.join(dfpop)



# Get the case numbers per 1 million population for each country  

df2['Confirmed/Pop'] = (df2['Confirmed'] / df2['Population']) * 1000000

df2['Recovered/Pop'] = (df2['Recovered'] / df2['Population']) * 1000000

df2['Deaths/Pop'] = (df2['Deaths'] / df2['Population']) * 1000000



df2.head()
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import calendar

import datetime

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



def n2rstr(n):

    if n > 1000:

        r = str(round(n/1000,1)) + 'K'

    else:

        r = str(int(n))

    return r



def plot_nCov(ax, df,country, show_simple=False,show_diff=False):

    lastday = df.iloc[-1].name

    m_start = datetime.date(lastday.year,lastday.month, 1)

    m_start_i = len(df[df.index <  m_start])



    diff_tag =  (lambda show_diff : ' - diff.' if show_diff==True else '')(show_diff) 

    c=['lightslategrey'] * len(df.index)

    for i in range(len(df[df.index >=  m_start])):

        c[-(i+1)] = ('#1F77B4')

    # Plot the data

    bars = ax.bar(df.index, df['Confirmed'], alpha=0.4, align='center', linewidth=0, color=c, label=f"Confimed {diff_tag} ({int(df.iloc[-1]['Confirmed']):,})")

    line_r = ax.plot(df['Recovered'], '-o', alpha=0.5, color='#1F77BF',label=f"Recovered {diff_tag} ({int(df.iloc[-1]['Recovered']):,})")

    line_d = ax.plot(df['Deaths'], '-^', alpha=0.5, color='#FF7777', label=f"Deaths {diff_tag} ({int(df.iloc[-1]['Deaths']):,})")



    # Show the legend 

    ax.legend(title=f"Current Numbers by Cases\n(As of {lastday})")

    

    # Set X Values

    ax.set_xticks(df.index)

    if show_simple:

        xlabels = len(df.index.values)*['']

    else:

        xlabels = list(pd.to_datetime(v).day for v in df.index.values)



    # Show Year and Month for the first and last day of the plot

    # and the start day of the current month.

    l_xlabel_full = lambda v : f"{v.day}\n{calendar.month_abbr[v.month]}\n{v.year} "

    xlabels[0] = l_xlabel_full(df.index.values[0])

    xlabels[m_start_i ] = l_xlabel_full(df.index.values[m_start_i]) 

    xlabels[-1] = l_xlabel_full(df.index.values[-1])         

    

    # Set the color for the x labels

    ax.set_xticklabels(xlabels, alpha=0.6)

    for xtick, color in zip(ax.get_xticklabels(), c):

        xtick.set_color(color)

        xtick.set_weight("bold")



    # Set the Title 

    ax.set_title(f'COVID-19 Confirmed in {country} {diff_tag}', alpha=0.8,fontsize=16)

    

    # Set Y values 

    ymax = ax.get_ylim()[1]

    ypad = ymax/30

    ax.set_ylim(bottom=-2*ypad, top=ymax + 1*ypad)

    ax.set_ylabel('Numbers of Case', alpha=0.8)



    if show_simple:

        x = bars[-1].get_x() + bars[-1].get_width()/2

        h = bars[-1].get_height()

        y_r = int(df['Recovered'][-1])

        y_d = int(df['Deaths'][-1])



        ax.text(x, h + ypad, n2rstr(h) , ha='center', color='lightslategrey', fontsize=9)   

        ax.text(x, y_r + ypad*0.5, n2rstr(y_r), fontsize=9, ha='center', color='#1F77BF', alpha=1.0)

        ax.text(x, y_d - ypad * 1.5, n2rstr(y_d), fontsize=9, ha='center', color='#FF7777', alpha=1.0)



    else:

        y_p = - 1 

        for bar in bars:

            x = bar.get_x() + bar.get_width()/2

            h = bar.get_height()

            y, c = (lambda v: [h + ypad,'lightslategrey'] if v else [h - ypad,'w'])(h < 3*ypad)

            # Do not show the number if it is the same as the previous one

            if y != y_p:    

                ax.text(x, y, n2rstr(h), ha='center', color=c, fontsize=9)   

                y_p = y



        y_r_p = 0

        y_d_p = 0

        for i in range(len(df)):

            x = df.index[i]

            y_r = int(df['Recovered'][i])

            y_d = int(df['Deaths'][i])



            # Do not show the number if it is the same as the previous one

            if y_r_p != y_r :

                ax.text(x, y_r + ypad*0.5, n2rstr(y_r), fontsize=9, ha='center', color='#1F77BF', alpha=0.8)

                y_r_p = y_r

            

            # Do not show the number if it is the same as the previous one

            if y_d_p != y_d :

                ax.text(x, y_d - ypad * 1.5, n2rstr(y_d), fontsize=9, ha='center', color='#FF7777', alpha=0.8)

                y_d_p = y_d



    # remove all the ticks (both axes), and tick labels on the Y axis

    ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)



    # remove the frame of the chart

    for spine in ax.spines.values():

        spine.set_visible(False)

        

    # And remove the grid

    ax.grid(False)

    

    # Watermarking by Country Name

    ax.text(df.index[int(len(df.index)/8)], ymax - ypad*15, country, fontsize=24, ha='left', color='#1F77BF', alpha=0.2)

    



def plot_nCov_ctry(ax, country, show_simple=False):

    dfctry = df[df['Country']== country]    

    # Set 'Date' as the index

    dfctry = dfctry.set_index('Date')

    plot_nCov(ax, dfctry, country, show_simple)





def plot_bar(ax, x_values, y_values, x_emp, case_name):

    bars = ax.bar(x_values, y_values, alpha=0.4, align='center', linewidth=0, color='lightslategrey')

    bars[x_emp].set_color('#1F77B4')

    

    # remove all the ticks (both axes), and tick labels on the Y axis

    ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)



    # remove the frame of the chart

    for spine in ax.spines.values():

        spine.set_visible(False)



    ax.grid(False)



    ax.legend()

    ax.set_title(f"Top 10 Countries of {case_name} Cases by Popluation\n(Cases per Millions)",fontsize=16)



    ax.get_xticklabels()[x_emp].set_color('#1F77B4')

    

    # Set Y values 

    ymax = ax.get_ylim()[1]

    ypad = ymax/30



    ax.set_ylim(bottom=-2*ypad, top=ymax + 1*ypad)

    ax.set_ylabel(f'{case_name} Cases per Population', alpha=0.8)



    for i in range(len(bars)):

        y = bars[i].get_height()

        ylabel = round(y,1)

        if i == x_emp : 

            c = '#1F77B4'

        else:

            c = 'lightslategrey'

        ax.text(bars[i].get_x() + bars[i].get_width()/2, y + ypad,ylabel , ha='center', color=c, fontsize=8)   

    plt.xticks(rotation=45)

fig = plt.figure(figsize=(20,25))

fig.patch.set_facecolor('#FFFFFF')

gspec = gridspec.GridSpec(3, 2, hspace=0.5)



fig.suptitle(f"COIVID-19 Data analysis\nBetween {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}\nusing data from Kaggle", fontsize=20)



plot_nCov_ctry(plt.subplot(gspec[0, :]), 'South Korea')



dft10 = df2['Confirmed/Pop'].sort_values(ascending=False)[:10]

plot_bar(plt.subplot(gspec[1, 0]), dft10.index, dft10,dft10.index.get_loc('South Korea'),'Confirmed')

dft10 = df2['Deaths/Pop'].sort_values(ascending=False)[:10]

plot_bar(plt.subplot(gspec[1, 1]), dft10.index, dft10 ,dft10.index.get_loc('South Korea'),'Deaths')





plot_nCov_ctry(plt.subplot(gspec[2, 0]), 'China',True)

plot_nCov(plt.subplot(gspec[2, 1]), df.groupby('Date')['Confirmed','Recovered','Deaths'].sum(), 'World Wide', True)

def plot_nCov_ctry_diff(ax, country, show_simple=False):

    dfctry = df[df['Country']== country]

    # Set 'Date' as the index

    dfctry = dfctry.set_index('Date')[cols].diff().fillna(0).astype(int)

    plot_nCov(ax, dfctry, country, show_simple, True)





    

fig = plt.figure(figsize=(20,15))

fig.patch.set_facecolor('#FFFFFF')



gspec = gridspec.GridSpec(2, 2, hspace=0.5)



fig.suptitle(f"COIVID-19 Data analysis\nBetween {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}\n(Daily Difference)", fontsize=20)



plot_nCov_ctry_diff(plt.subplot(gspec[0, :]), 'South Korea')

plot_nCov_ctry_diff(plt.subplot(gspec[1, 0]), 'China',True)

df_world =  df[df['Country']!='China'].groupby('Date')['Confirmed','Recovered','Deaths'].sum()

df_world_diff = df_world[cols].diff().fillna(0).astype(int)

plot_nCov(plt.subplot(gspec[1, 1]), df_world_diff, 'World Wide (Excluding China)', True,True)

from IPython.core.display import display, HTML



def print_num(num):

    if num - int(num) > 0 :

        r = '{:.4f}'.format(num)

    else:

        r = f'{int(num):,}'

    return r



def show_nCov_ww_title(df, world, title):    

    display(HTML(f'<h1>{title}</h1><h2>(updated: {updated})</h2>'))

    

    row_fmt = '''

    <tr>

        <th>{}</th>

        <td bgcolor= "white"><b><font color="blue">{}</b></font></td>

        <td bgcolor= "white"><b><font color="green">{}</b></font></td>

        <td bgcolor= "white"><b><font color="red">{}</b></font></td>

    </tr>

    '''

    html_rows = row_fmt.format('Worldwide', print_num(world[0]), print_num(world[1]), print_num(world[2]))



    df_rank = df.sort_values(cols[0],ascending=False)

    for i in range(10):

        row = df_rank.iloc[i]

        html_rows = html_rows + row_fmt.format(row.name,print_num(row[0]),print_num(row[1]),print_num(row[2]) )   

 

    display(HTML(f'''

    <table>

       <caption># of COVID-19 cases as of {updated}</caption>

      <tr>

        <th></th>

        <th bgcolor="blue"><font color="white">Confirmed</font></th>

        <th bgcolor="green"><font color="white">Recovered</font></th>

        <th bgcolor="red"><font color="white">Deaths</font></th>

      </tr>

      {html_rows}

    </table>

    '''))
import matplotlib.pyplot as plt

import mplleaflet



cols = ['Confirmed','Recovered','Deaths']

world= df2[cols].sum().to_list()

show_nCov_ww_title(df2[cols],world, '# of COVID-19 cases per country population')



lons = df2['Long'].tolist()

lats = df2['Lat'].tolist()



fig = plt.figure(figsize=(8,8))

plt.scatter(lons, lats, c='b',alpha=0.4, s=df2[cols[0]]);

plt.scatter(lons, lats, c='g',alpha=0.4, s=df2[cols[1]]);

plt.scatter(lons, lats, c='r',alpha=0.4, s=df2[cols[2]]);



mplleaflet.display(fig=fig,tiles='cartodb_positron')
import matplotlib.pyplot as plt

import mplleaflet



cols = ['Confirmed/Pop','Recovered/Pop','Deaths/Pop']

w = df2[cols + ['Population']].sum()

world = [w[0]/w[3], w[1]/w[3], w[2]/w[3]]

show_nCov_ww_title(df2[cols],world, '# of COVID-19 Cases by Popluation(Cases per Millions)')



lons = df2['Long'].tolist()

lats = df2['Lat'].tolist()



fig = plt.figure(figsize=(8,8))

plt.scatter(lons, lats, c='b',alpha=0.4, s=df2[cols[0]]*100);

plt.scatter(lons, lats, c='g',alpha=0.4, s=df2[cols[1]]*100);

plt.scatter(lons, lats, c='r',alpha=0.4, s=df2[cols[2]]*100);



mplleaflet.display(fig=fig,tiles='cartodb_positron')
import seaborn as sns; sns.set()

import matplotlib.pyplot as plt



df_world = df.groupby('Date').sum()

df_china = df[df['Country'] == 'China'].set_index('Date')[['Confirmed','Deaths','Recovered']]

df_cw = (df_china / df_world)*100



plt.figure(figsize=(15,5))

plt.xticks(rotation=45)

ax = sns.lineplot(data=df_cw)

plt.title("Numbers of China among the cases worldwide (%)", fontsize=14);