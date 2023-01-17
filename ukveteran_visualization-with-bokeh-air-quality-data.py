import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bokeh.io import output_file,show,output_notebook,save

from bokeh.plotting import figure
df = pd.read_csv("../input/air-quality-data/airmay.csv")

df
fig = figure(x_axis_label = "X1",y_axis_label = "X2")
fig.circle(df["X1"],df["X2"])
output_notebook()
show(fig)
fig.line(df["X1"],df["X2"])
show(fig)
fig = figure(x_axis_label = "X1",y_axis_label = "X3")
fig.circle(df["X1"],df["X3"])
output_notebook()
show(fig)
fig.line(df["X1"],df["X3"])
show(fig)