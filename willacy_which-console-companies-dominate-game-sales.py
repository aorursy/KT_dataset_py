# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/vgsales.csv")

df = df[~df.Year.isnull()] # drop all rows with empty year values

df.Year = df.Year.astype(int)

df.shape
df.head()
firm_key = {

                'DS'   : 'Nintendo',

                'PS2'  : 'Sony',

                'PS3'  : 'Sony',

                'Wii'  : 'Nintendo',

                'X360' : 'Microsoft',

                'PSP'  : 'Sony',

                'PS'   : 'Sony',

                'PC'   : 'Microsoft',

                'XB'   : 'Microsoft',

                'GBA'  : 'Nintendo',

                'GC'   : 'Nintendo',

                '3DS'  : 'Nintendo',

                'PSV'  : 'Sony',

                'PS4'  : 'Sony',

                'N64'  : 'Nintendo',

                'SNES' : 'Nintendo',

                'XOne' : 'Microsoft',

                'SAT'  : 'Sega',

                'WiiU' : 'Nintendo',

                '2600' : 'Atari',

                'GB'   : 'Nintendo',

                'NES'  : 'Nintendo',

                'DC'   : 'Sega',

                'GEN'  : 'Sega',

                'NG'   : 'Nokia',

                'WS'   : 'Bandai',

                'SCD'  : 'Sega',

                '3DO'  : 'Panasonic',

                'TG16' : 'NEC',

                'PCFX' : 'NEC',

                'GG'   : 'Sega'

} 





console_type_key = {

                'DS'   : 'Handheld',

                'PS2'  : 'Console',

                'PS3'  : 'Console',

                'Wii'  : 'Console',

                'X360' : 'Console',

                'PSP'  : 'Handheld',

                'PS'   : 'Console',

                'PC'   : 'PC',

                'XB'   : 'Console',

                'GBA'  : 'Handheld',

                'GC'   : 'Console',

                '3DS'  : 'Handheld',

                'PSV'  : 'Handheld',

                'PS4'  : 'Console',

                'N64'  : 'Console',

                'SNES' : 'Console',

                'XOne' : 'Console',

                'SAT'  : 'Console',

                'WiiU' : 'Console',

                '2600' : 'Console',

                'GB'   : 'Handheld',

                'NES'  : 'Console',

                'DC'   : 'Console',

                'GEN'  : 'Console',

                'NG'   : 'Handheld',

                'WS'   : 'Handheld',

                'SCD'  : 'Console',

                '3DO'  : 'Console',

                'TG16' : 'Console',

                'PCFX' : 'Console',

                'GG'   : 'Handheld'

} 



df['Platform_Firm'] = df.Platform.map(firm_key)

df['Platform_Type'] = df.Platform.map(console_type_key)
plt.rc('font', size=18)

fig = plt.figure(figsize=(18,9))



df[df.Platform_Type == 'Console'].groupby('Year')['Global_Sales'].sum().plot()

df[df.Platform_Type == 'Handheld'].groupby('Year')['Global_Sales'].sum().plot()

df[df.Platform_Type == 'PC'].groupby('Year')['Global_Sales'].sum().plot()



plt.legend(['Console','Handheld','PC'], loc='best')

plt.title("Global Sales for Games Sold")

plt.ylabel("Sales (in millions)")
plt.rc('font', size=13)

fig = plt.figure(figsize=(18,9))



platform_firm = list(df.Platform_Firm.value_counts().index) # List of Firms i.e. Sony, Nintendo.



ax1 = plt.subplot2grid((2,2),(0,0))

for firm in platform_firm:

    # Sum up the Global Sales for console made games per Platform Firms

    ax = df[df.Platform_Type == 'Console']

    # Try to plot data for each year per firm. Some Firms like Atari have no sales to plot for certain years.

    try:

        ax[(ax.Platform_Firm == firm)].groupby(['Year'])['Global_Sales'].sum().plot()

        plt.legend(platform_firm, loc='best')

        ax1.set_title("Global Sales for Console Games Sold")

        ax1.set_ylabel("Sales (in millions)")

    except TypeError:

        continue



ax2 = plt.subplot2grid((2,2),(0,1))

for firm in platform_firm:

    # Same as above but for handheld.

    # Should be able to reduce code here by looping through handheld and console.

    ax = df[df.Platform_Type == 'Handheld']

    try:

        ax[(ax.Platform_Firm == firm)].groupby(['Year'])['Global_Sales'].sum().plot()

        plt.legend(platform_firm, loc='best')

        ax2.set_title("Global Sales for Handheld Games Sold")

        ax2.set_ylabel("Sales (in millions)")

    except TypeError:

        continue

        

# Plot PC games seperately

# Could have put PC with Consoles but was interested in being able to compare them.

ax3 = plt.subplot2grid((2,2),(1,0))

df[df.Platform_Type == 'PC'].groupby(['Year'])['Global_Sales'].sum().plot(label='PC')

plt.legend('PC', loc='best')

ax3.set_title("Global Sales for PC Games Sold")

ax3.set_ylabel("Sales (in millions)")



# Comparing all games between firms, regardless of type.

ax4 = plt.subplot2grid((2,2),(1,1))

for firm in platform_firm:

    try:

        df[(df.Platform_Firm == firm)].groupby(['Year'])['Global_Sales'].sum().plot()

        plt.legend(platform_firm, loc='best')

        ax4.set_title("Global Sales for Handheld Games Sold")

        ax4.set_ylabel("Sales (in millions)")

    except TypeError:

        continue

        

plt.tight_layout()
global_sales = df.groupby('Year')['Global_Sales'].sum()



# Create a Series for handheld, console and PC with year and sales as a fraction of total sales for that year.

console_sales_percent = (df[df.Platform_Type == 'Console'].groupby('Year')['Global_Sales'].sum()/global_sales)

handheld_sales_percent = (df[df.Platform_Type == 'Handheld'].groupby('Year')['Global_Sales'].sum()/global_sales)

pc_sales_percent = (df[df.Platform_Type == 'PC'].groupby('Year')['Global_Sales'].sum()/global_sales)



# Create DataFrame with the years as the index and the sales numbers for each platform type as a new column

# This will allow us to use the stacked variable when plotting.

game_type_df = pd.DataFrame({'Console': console_sales_percent.values,

                            'Handheld': handheld_sales_percent.values,

                            'PC': pc_sales_percent.values},

                            index = console_sales_percent.index)



plt.rc('font', size=18)

ax10 = game_type_df.plot(kind='bar', stacked=True, width=1.0, figsize=(15,8))

vals = ax10.get_yticks()

# Change the y values to a percent.

ax10.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])

plt.title("Percentage of Global Sales made between Console, Handheld, and PC")

plt.xlabel("Year")

plt.ylabel("Percent")

plt.ylim(0,1)

# Print legend outside of the graph

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
firm_df_key = {}



# Create Dictionary with Firm name as key sales per year in a Series as the value. 

# i.e. {'Atari': Year

#                1980    11.38

#                1981    35.77...

for i in list(df.Platform_Firm.value_counts().index):

    firm_df_key[i] = df[df.Platform_Firm == i].groupby('Year')['Global_Sales'].sum()



# Create empty DataFrame with index as the years.

firm_percent_df = pd.DataFrame(index=df.Year.value_counts().index.sort_values())



# Function takes a firm like Sony, and a Date like 1980

# makes a dictionary(dic) of the Series made in the firm_df_key for that firm.

# If the date is in that Series/dic it returns the sales value, otherwise theres no sales so returns 0.

def getit(date, firm):

    dic = firm_df_key[firm].to_dict() # dic = {1980: 0.25552, 1981: 0.3255} etc.

    if date in dic:

        return dic[date]/(df.groupby('Year')['Global_Sales'].sum().to_dict())[date]

    else:

        return 0



# Creates a new column on the new dataframe for each firm with sales per year as a fraction

for firm in list(df.Platform_Firm.value_counts().index):

    firm_percent_df[firm] = firm_percent_df.index.map(lambda date: getit(date, firm))  #i.e.1980   Nokia



    

plt.rc('font', size=18)

ax11 = firm_percent_df.plot(kind='bar', stacked=True, width=1.0, figsize=(15,8))

yvals = ax11.get_yticks()

ax11.set_yticklabels(['{:3.0f}%'.format(x*100) for x in yvals])

plt.ylim(0,1)

plt.ylabel("Percent")

plt.title("Percentage of Global Profits for Games Sold per Gaming Firm")

plt.xlabel('Year')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)