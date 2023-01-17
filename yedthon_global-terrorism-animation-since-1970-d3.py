import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from matplotlib.ticker import AutoMinorLocator, FuncFormatter

import datetime

import gzip
%matplotlib inline

plt.rcParams["axes.labelsize"] = 30

sns.set(font_scale=1.8)
from IPython.display import display_html

display_html("""<button onclick="$('.input, .prompt, .output_stderr, .output_error').toggle();">Toggle Code</button>""", raw=True)
df = pd.read_csv("../input/globalterrorismdb_0617dist.csv", encoding='ISO-8859-1')
# df.columns.values
df.shape
dfs = df[["eventid", "iyear", "imonth", "iday",

          "country_txt", "city", "latitude","longitude",

          "success", "suicide", "attacktype1_txt", "targtype1_txt", "weaptype1_txt",

          "gname", "motive", "summary", "nkill", "nwound"]]
dfs = dfs.rename(

    columns={"eventid": "id", 'iyear':'year', 'imonth':'month', 'iday':'day', 'country_txt':'country',

             'attacktype1_txt':'attacktype', 'targtype1_txt':'target', 'weaptype1_txt':'weapon',

             'gname': 'group', 'nkill':'fatalities', 'nwound':'injuries'})

dfs['fatalities'] = dfs['fatalities'].fillna(0).astype(int)

dfs['injuries'] = dfs['injuries'].fillna(0).astype(int)
dfs['country'].replace('United States', 'United States of America',inplace=True)



dfs['country'].replace('Bosnia-Herzegovina', 'Bosnia and Herzegovina',inplace=True)



dfs['country'].replace('Slovak Republic', 'Slovakia',inplace=True)
len(set(dfs["country"]))
len(dfs[dfs["fatalities"] != 0]) / len(dfs)
len(dfs[dfs["injuries"] != 0]) / len(dfs)
len(dfs[(dfs["injuries"] != 0) | (dfs["fatalities"] != 0)]) / len(dfs)
dfs['fatalities'].sum()
dfs['injuries'].sum()
dfs_map = dfs[['id', 'year', 'latitude', 'longitude',

               'attacktype', 'target', 'summary', 'fatalities', 'injuries']]



dfs_map = dfs_map.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'attachtype': 'attack'})
dfs_map['intensity'] = dfs_map['fatalities'] + dfs_map['injuries'] + 1



dfs_map['id'] = dfs_map.index
dfs_map = dfs_map[dfs_map["lat"].notnull() & dfs_map["lon"].notnull()]



dfs_map = dfs_map[(dfs_map["lon"] >= -180) & (dfs_map["lon"] <= 180) & (dfs_map["lat"] >= -90) & (dfs_map["lat"] <= 90)]
len(dfs_map)
# dfs_map.to_json("./static/data/global_terrorism_map.json", orient='records')
dfs_map['intensity'].max()
def freq_table(df, column):

    frequency_table = pd.crosstab(index=df[column],

                columns="count").sort_values("count", ascending=False)

    return frequency_table
fig, (ax1, ax2) = plt.subplots(2, figsize=(3, 5),  sharex=True)

n = 0

for i in ["success", "suicide"]:      

    n += 1

    ax =  locals().get('ax' + str(n))

    frequency_table = freq_table(dfs, i)

#     if "Unknown" in frequency_table.index:

#         frequency_table = frequency_table.drop("Unknown") .iloc[:20]

#     else:

#         frequency_table = frequency_table.iloc[:20]

    

    sns.barplot(y="count", x=frequency_table.index, data=frequency_table, color="grey", ax=ax)
# fig, (ax1, ax2, ax3) = plt.subplots(figsize=(16, 9), nrows=3)

for i in ["country", "city", "group"]:    

    fig, ax = plt.subplots(figsize=(16, 9))

    ax.set_title(i, fontsize= 30)

    frequency_table = freq_table(dfs, i)

    if "Unknown" in frequency_table.index:

        frequency_table = frequency_table.drop("Unknown") .iloc[:20]

    else:

        frequency_table = frequency_table.iloc[:20]

    

    sns.barplot(y=frequency_table.index, x="count", data=frequency_table, orient="h", color="grey")
# fig, (ax1, ax2, ax3) = plt.subplots(figsize=(16, 9), nrows=3)

for i in ["attacktype", "target", "weapon"]:    

    fig, ax = plt.subplots(figsize=(16, 3))

    ax.set_title(i, fontsize= 30)

    frequency_table = freq_table(dfs, i)

    frequency_table = frequency_table.iloc[:5]

    

    sns.barplot(y=frequency_table.index, x="count", data=frequency_table, orient="h", color="grey")
dfs_time = dfs.copy()



dfs_time = dfs_time[(dfs_time["month"] >=1) & (dfs_time["month"] <= 12) & (dfs_time["day"] >=1) & (dfs_time["day"] <= 31)]



dfs_time["date"] = pd.to_datetime(dfs_time[["year", "month", "day"]])
# dfs_time
for i in ["attacktype", "target", "weapon"]:

    top = freq_table(dfs, i)[:5].index.values

    dfs_ts= dfs_time[dfs_time[i].isin(top)]

    dfs_ts = dfs_ts.groupby(["date", i])[i].count()

    dfs_ts = dfs_ts.reset_index(level=1, name="count")

    dfs_ts = dfs_ts.groupby([pd.TimeGrouper(freq='Q'), i])["count"].sum()

    dfs_ts = dfs_ts.reset_index()

    fig, ax = plt.subplots(figsize=(16, 9))

    # assign locator and formatter for the xaxis ticks.

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: pd.to_datetime(x).year))

    # ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

    fig.autofmt_xdate()

    sns.tsplot(dfs_ts,

               time="date", value="count", unit=i, condition=i,

              color=sns.color_palette("Set1"))
fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 9), sharex=True)

n = 0

for i in ["success", "suicide"]:      

    n += 1

    ax =  locals().get('ax' + str(n))

    

    dfs_ts = dfs_time.groupby(["date", i])[i].count()

    dfs_ts = dfs_ts.reset_index(level=1, name="count")

    dfs_ts = dfs_ts.groupby([pd.TimeGrouper(freq='Q'), i])["count"].sum()

    dfs_ts = dfs_ts.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))

    

    dfs_ts = dfs_ts.reset_index()

    dfs_ts = dfs_ts[dfs_ts[i] == 1]

    

    ax.xaxis.set_major_locator(mdates.YearLocator(5))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.xaxis.set_minor_locator(AutoMinorLocator())



    datemin = datetime.date(dfs_ts.date.min().year, 1, 1)

    datemax = datetime.date(dfs_ts.date.max().year + 1, 1, 1)

    ax.set_xlim(datemin, datemax)  

    ax.axvline("2001-09-11", color="black", linestyle="dashed", alpha=0.6)

    ax.axvline("2007-08-06", color="black", linestyle="dashed", alpha=0.6)

    ax.axvline("2008-09-15", color="black", linestyle="dashed", alpha=0.6)



    if n == 1:

        ax.text("2008-09-15", 88, "Lehman Brothers collapse",

                verticalalignment='center',

                rotation=90, size=20)        

    if n == 2:        

        ax.text("2001-09-11", 10, "911", verticalalignment='center', size=20)

        ax.text("2007-08-06", 5, "AHMI collapse",

                verticalalignment='center', horizontalalignment='right',

                rotation=90, size=20)



    

    ax.plot(dfs_ts["date"], dfs_ts["count"])

    

#     sns.tsplot(dfs_ts,

#            time="date", value="count", unit=i, condition=i,

#           color=sns.color_palette("Set2"), ax=ax)

#     ax.legend_.remove()    

    

    ax.set_title(i + " attack percent", size=20)



    fig.autofmt_xdate()
dfs_period_casualties = dfs_time.groupby(["date"])["fatalities", "injuries"].sum()



dfs_period_casualties = dfs_period_casualties.groupby(pd.TimeGrouper(freq='M'))["fatalities", "injuries"].sum()



dfs_period_casualties = dfs_period_casualties.reset_index()
fig, ax = plt.subplots(figsize=(16, 9))

# ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.plot(dfs_period_casualties["date"], dfs_period_casualties["fatalities"], color='xkcd:red')

ax.plot(dfs_period_casualties["date"], dfs_period_casualties["injuries"], color='xkcd:orange')

ax.xaxis.set_major_locator(mdates.YearLocator(5))

# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax.xaxis.set_minor_locator(AutoMinorLocator())

datemin = datetime.date(dfs_period_casualties.date.min().year, 1, 1)

datemax = datetime.date(dfs_period_casualties.date.max().year + 1, 1, 1)

ax.set_xlim(datemin, datemax)

fig.autofmt_xdate()



ax.legend(prop={'size': 20})



plt.axvline(x="2001-09-11", color="black", linestyle="dashed", alpha=0.6)

plt.text("2001-09-11", 14000, "911", verticalalignment='center', size=20)



plt.axvline(x="1995-03-20", color="black", linestyle="dashed", alpha=0.6)

plt.text("1995-03-20", 8000, "Tokyo subway sarin attack", rotation=90,

         verticalalignment='center', horizontalalignment='right', size=20)



plt.axvline(x="1998-08-07", color="black", linestyle="dashed", alpha=0.6)

plt.text("1998-08-07", 8000, "1998 United States embassy bombings",

         horizontalalignment='right', verticalalignment='center',

        rotation=90, size=20)



plt.axvline(x="2011-12-18", color="black", linestyle="dashed", alpha=0.6)

plt.text("2011-12-18", 8000, "Last U.S. troops withdrew from Iraq",

         horizontalalignment='right', verticalalignment='center',

        rotation=90, size=20)



plt.xlabel('Year', size=20)

plt.ylabel('Number', size=20)

plt.title('Casualities in Terrorism', size=30)