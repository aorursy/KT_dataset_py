import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from bokeh.plotting import figure, output_notebook, show, vplot, ColumnDataSource
from bokeh.charts import TimeSeries
from bokeh.models import HoverTool, CrosshairTool
from bokeh.palettes import brewer
import gc
import dask.dataframe as dd

output_notebook()
train  =  pd.read_csv('../input/train.csv', usecols = ('date_time', 'hotel_cluster','is_booking'), 
                      parse_dates = ['date_time'])
train['dow'] = train.date_time.dt.weekday
train['year'] = train.date_time.dt.year
train['month'] = train.date_time.dt.month
train['day'] = train.date_time.dt.day
train_agg = train.groupby(['dow','year','month','day', 'hotel_cluster']).agg(['sum', 'count'] )
train_agg.columns = ('bookings', 'total')
train_agg.head()

del(train)

gc.collect()
pv_agg = train_agg.reset_index()
pv_agg['dt'] = pd.to_datetime( pv_agg.year*10000 + pv_agg.month*100 + pv_agg.day
                                  , format='%Y%m%d')
pv_agg = pv_agg.pivot(index = 'dt', columns = 'hotel_cluster', values = 'bookings')
pv_agg.columns = [str(i) for i in pv_agg.columns]
pv_agg['dt'] = pv_agg.index
pv_agg['dow'] = pv_agg.dt.dt.weekday
pv_agg.head()