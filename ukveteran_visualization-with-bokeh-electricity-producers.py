import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bokeh.io import output_file,show,output_notebook,save

from bokeh.plotting import figure
df = pd.read_csv("../input/cost-function-for-electricity-producers/Electricity.csv")

df
fig = figure(x_axis_label = "cost",y_axis_label = "q")
fig.circle(df["cost"],df["q"])
output_notebook()
show(fig)
fig.line(df["cost"],df["q"])
show(fig)
fig.x(df["cost"],df["pl"],line_color="red",size=10)
show(fig)
fig.line(df["cost"],df["pl"],line_color="red")
show(fig)