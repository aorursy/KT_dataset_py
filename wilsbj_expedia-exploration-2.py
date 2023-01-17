# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

dest = pd.read_csv('../input/destinations.csv',
                    #dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32},
                    #usecols=['srch_destination_id','is_booking','hotel_cluster'],
                    #chunksize=1000000
                   )
dest

train = pd.read_csv('../input/train.csv',
                    #dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32},
                    #usecols=['srch_destination_id','is_booking','hotel_cluster'],
                    chunksize=1000000)
aggs = []
print('-'*38)
for chunk in train:
    agg = chunk.groupby(['srch_adults_cnt'])['is_booking'].agg(['sum','count'])
    agg.reset_index(inplace=True)
    aggs.append(agg)
    print('.',end='')
print('')
aggs = pd.concat(aggs, axis=0)
aggs.head()
train = pd.read_csv('../input/train.csv'
                    ,nrows=100000
                    #,chunksize=100000
                   )
train
from matplotlib import pyplot as plt
plt.scatter(aggs['sum'][1:100],aggs['count'][1:100])
fit = np.polyfit(aggs['sum'][1:100],aggs['count'][1:100],1)
fit
plt.plot(x,y, 'yo', x, fit_fn(x), '--k')
plt.xlim(0, 5)
plt.ylim(0, 12)
test = pd.read_csv('../input/test.csv',
                    nrows = 100000
                    #dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32},
                    #usecols=['srch_destination_id','is_booking','hotel_cluster'],
                    #chunksize=1000000)
                  )
test.describe()
train.describe()
train.groupby('srch_adults_cnt')
plt.scatter(aggs['srch_adults_cnt'],aggs['sum']/aggs['count'])
plt.scatter(aggs['srch_adults_cnt'],aggs['sum']/aggs['count'])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
output_notebook()

from bokeh.plotting import figure, output_notebook, show, vplot, ColumnDataSource
from bokeh.charts import TimeSeries
from bokeh.models import HoverTool, CrosshairTool
from bokeh.palettes import brewer
import gc
import dask.dataframe as dd
train  =  pd.read_csv('../input/train.csv', usecols = ('date_time', 'srch_destination_type_id','is_booking'), 
                      parse_dates = ['date_time'])
train['dow'] = train.date_time.dt.weekday
train['year'] = train.date_time.dt.year
train['month'] = train.date_time.dt.month
train['day'] = train.date_time.dt.day
train_agg = train.groupby(['dow','year','month','day', 'srch_destination_type_id']).agg(['sum', 'count'] )
train_agg.columns = ('bookings', 'total')
train_agg.head()
del(train)

gc.collect()
pv_agg = train_agg.reset_index()
pv_agg['dt'] = pd.to_datetime( pv_agg.year*10000 + pv_agg.month*100 + pv_agg.day
                                  , format='%Y%m%d')
pv_agg = pv_agg.pivot(index = 'dt', columns = 'srch_destination_type_id', values = 'bookings')
pv_agg.columns = [str(i) for i in pv_agg.columns]
pv_agg['dt'] = pv_agg.index
pv_agg['dow'] = pv_agg.dt.dt.weekday
pv_agg.head()
def make_hc_plot(df, start, stop):
#    hover = HoverTool(
#        tooltips=[
#            ("Date", "@day"),
#            ("Day of week", "@dow"),
#            ("srch_destination_type_id", "@srch_destination_type_id"),
#            ("bookings", "@bookings"),
#        ]
#    )
    
    #colors = brewer['RdYlBu'][stop-start]
    colors = ['red', 'darkmagenta', 'green', 'darkorange', 'blue']
#    ch = CrosshairTool(dimensions = ['height'], line_color='red')
    p = figure(x_axis_type = 'datetime',plot_width=800, plot_height=400, tools=[hover, ch, 'pan,wheel_zoom,save,box_zoom,reset,resize'])
    p.title = 'Expedia bookings for subset of srch_destination_type_id {} to {}'.format(start, stop)
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Number of bookings'
    
    for i in range(start, stop):
        src  = ColumnDataSource({'day': df.dt.dt.strftime('%Y-%m-%d'), 
                                 'dow': df.dow.tolist(),
                                 'bookings': df[str(i)],
                                 'srch_destination_type_id': [i]*df.shape[0]})
        
        p.line((df['dt']), df[str(i)], color=colors[i-start], legend = str(i), source = src)

    return p

tslines = []

for i in range(0,9,3):
    tsline = make_hc_plot(pv_agg, i, i+3)
    tslines.append(tsline)
    
show(vplot(*tslines))

