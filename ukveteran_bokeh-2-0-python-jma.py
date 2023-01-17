from bokeh.io import output_file,show,output_notebook,push_notebook

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper

from bokeh.layouts import row,column,gridplot

from bokeh.models.widgets import Tabs,Panel

from bokeh.models import ColumnDataSource, HoverTool

from bokeh.sampledata.autompg import autompg_clean as df

from bokeh.transform import factor_cmap

output_notebook()
df.cyl = df.cyl.astype(str)

df.yr = df.yr.astype(str)



group = df.groupby(by=['cyl', 'mfr'])

source = ColumnDataSource(group)



p = figure(plot_width=800, plot_height=300, title="Mean MPG by # Cylinders and Manufacturer",

           x_range=group, toolbar_location=None, tools="")



p.xgrid.grid_line_color = None

p.xaxis.axis_label = "Manufacturer grouped by # Cylinders"

p.xaxis.major_label_orientation = 1.2



index_cmap = factor_cmap('cyl_mfr', palette=['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c'], 

                         factors=sorted(df.cyl.unique()), end=1)



p.vbar(x='cyl_mfr', top='mpg_mean', width=1, source=source,

       line_color="white", fill_color=index_cmap, 

       hover_line_color="darkgrey", hover_fill_color=index_cmap)



p.add_tools(HoverTool(tooltips=[("MPG", "@mpg_mean"), ("Cyl, Mfr", "@cyl_mfr")]))



show(p)
import pandas as pd

import numpy as np



np.random.seed(1)

num=10000



dists = {cat: pd.DataFrame(dict(x=np.random.normal(x,s,num),

                                y=np.random.normal(y,s,num),

                                val=val,cat=cat))

         for x,y,s,val,cat in 

         [(2,2,0.01,10,"d1"), (2,-2,0.1,20,"d2"), (-2,-2,0.5,30,"d3"), (-2,2,1.0,40,"d4"), (0,0,3,50,"d5")]}



df = pd.concat(dists,ignore_index=True)

df["cat"]=df["cat"].astype("category")

df.tail()



import datashader as ds

import datashader.transfer_functions as tf



%time tf.shade(ds.Canvas().points(df,'x','y'))
canvas = ds.Canvas(plot_width=250, plot_height=250, x_range=(-4,4), y_range=(-4,4))

agg = canvas.points(df, 'x', 'y', agg=ds.count())

agg
tf.shade(agg.where(agg>=np.percentile(agg,99)))
tf.shade(np.sin(agg))
tf.shade(agg, cmap=["darkred", "yellow"])
tf.shade(agg,cmap=["darkred", "yellow"],how='linear')
tf.shade(agg,cmap=["darkred", "yellow"],how='log')
tf.shade(agg,cmap=["darkred", "yellow"],how='eq_hist')
color_key = dict(d1='blue', d2='green', d3='red', d4='orange', d5='purple')

aggc = canvas.points(df, 'x', 'y', ds.count_cat('cat'))

tf.shade(aggc, color_key)
tf.spread(tf.shade(aggc, color_key))