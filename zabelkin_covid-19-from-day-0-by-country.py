# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
# extra columns for data unification
df['date'] = pd.to_datetime(df['Last Update']).dt.date
df['Active'] = df.Confirmed-df.Recovered-df.Deaths
# data focus
focus_field = 'Active' # could be also 'Confirmed', 'Recovered'or 'Deaths'

# country name and offset of "Day 0" from 1st data available
countries=dict(
    Russia=12,
    Italy=0,
    Spain=0,
    Germany=13,
    France=13,
    Iran=2,
    Turkey=1,
    Moscow=0
  )
# colelcting an array of Series rather then DataFrame 
# as observation duration may vary by country at the beggining of pandemic
output_series = []
for _cntr, _offset in countries.items():
    if _cntr=='Moscow':
        # consider Moscow as dedicated region
        #output_series += df[df['Province/State']=="Moscow"].reset_index().loc[_offset:,['date', focus_field]]
        a = df[df['Province/State']==_cntr].groupby('date')[focus_field].sum()[_offset:].reset_index()
        # as observations for region has started 72 after Russia overall numbers 
        a.index +=72
        output_series += [a]
    else:
        output_series += [df[df['Country/Region']==_cntr].groupby('date')[focus_field].sum()[_offset:].reset_index()]
# some data anomalities hand repared with hardcoding :)
def smooth(series_no, bad_index):
    return (
        output_series[series_no].loc[bad_index-1, focus_field] + 
        output_series[series_no].loc[bad_index+1, focus_field])/2

output_series[1].loc[18] = smooth(1, 18)
output_series[1].loc[21] = smooth(1, 21)
output_series[1].loc[22] = smooth(1, 22)
output_series[2].loc[15] = smooth(2, 15)
output_series[3].loc[12] = smooth(3, 12)
output_series[4].loc[11] = smooth(4, 11)
output_series[5].loc[16] = smooth(5, 16)

# Spain stopped to report recovered at about day 155
output_series[2].drop(range(155, len(output_series[2]),1), inplace=True)

# France sieze to report recovered at about day 165
output_series[4].drop(range(165, len(output_series[4]),1), inplace=True)


# Russian data as of today, if can't wait for DB update
# source https://www.worldometers.info/coronavirus/ or media ----------------V          this number to change
# output_series[0].loc[len(output_series[0])]=[pd.Timestamp.today().date(),243868]
# graphing

# annotation proc
def graph_ann(in_data, _ax):
    x_ax = in_data.index[len(in_data)-1]
    y_ax = in_data.index[len(in_data.index)-1] # taking last number index array
    _=_ax.annotate("{:,}".format(int(in_data.loc[y_ax,focus_field])),\
              xy=(x_ax, in_data.loc[y_ax,focus_field]*0.96))

# limites by day of observations
min_x = 0
DAYS_IN_TICK = 10

max_x = round(max([len(ser_len) for ser_len in output_series])/DAYS_IN_TICK)*DAYS_IN_TICK+DAYS_IN_TICK
is_log = False # logarithmic scale for axe Y?


for _idx, _cntr in enumerate(countries.keys()):
    #initiating .plot method for 1st Series then just adding data
    if (_idx==0):
        ax = (output_series[0].reset_index().plot(
                x='index',
                y=focus_field, figsize=(15,9), linewidth = 4,
                xticks=range(min_x,max_x,DAYS_IN_TICK), xlim = (min_x, max_x),
                logy=is_log,
                colormap='Paired',
                label=_cntr, title=focus_field + " cases on "+df.date.max().strftime("%d.%m.%Y") )
             )
        ax.set
    else:
        ax = output_series[_idx].reset_index().plot(x='index',y=focus_field, ax=ax, xlim = (min_x, max_x), label=_cntr, linewidth = 3)
    graph_ann(output_series[_idx], ax)
    #extra mark for my home country 
    if _cntr=='Russia':
        ax.axhline(y=int(output_series[0].loc[len(output_series[0])-1,focus_field]),
                   color='red', linestyle='--', linewidth = 1)

# end of cycle, adding other elements

annotations_list = [
    ['Moscow closes\n2020-04-01', '2020-04-01'],
    ['Non-working days end\n2020-05-11', '2020-05-11'],
    ['Moscow opens\n2020-06-01', '2020-06-01'],
    ['Parade\n2020-06-24', '2020-06-24'],
    ['Constitution censys\n2020-07-01', '2020-07-01'],
]

# annotating events in Russia
for _idx, _annot in enumerate(annotations_list):
    x = output_series[0][output_series[0].date==pd.to_datetime(_annot[1])].index[0]
    y = output_series[0][output_series[0].date==pd.to_datetime(_annot[1])].iloc[0, 1]
    ax.annotate(_annot[0], xy=(x,y), xytext=(x,175000+25000*(_idx%2)),
                arrowprops=dict(color='red', width=1, headwidth=10, headlength=10, shrink=0.04),
                 horizontalalignment='center',
               )

# extra decorations
ax.legend(loc='center left')#, bbox_to_anchor=(1.2, 1))
ax.grid(color='grey', linestyle='-.', linewidth=0.5)
ax.set_xlabel("days of observation")
ax.set_xticks(range(min_x,max_x,DAYS_IN_TICK))
ax.figure.set_facecolor('white')

#output_series[2].drop(range(151,182,1))