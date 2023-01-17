import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bokeh.io import output_file,show,output_notebook,save

from bokeh.plotting import figure
df = pd.read_csv("../input/the-joynerboore-attenuation-data/attenu.csv")

df
fig = figure(x_axis_label = "dist",y_axis_label = "accel")
fig.circle(df["dist"],df["accel"])
output_notebook()
show(fig)
fig.line(df["dist"],df["accel"])
show(fig)
fig.x(df["dist"],df["mag"],line_color="red",size=10)
show(fig)
fig.line(df["dist"],df["mag"],line_color="red")
show(fig)