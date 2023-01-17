import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bokeh.io import output_file,show,output_notebook,save

from bokeh.plotting import figure
df = pd.read_csv("../input/wages-data/Griliches.csv")

df
fig = figure(x_axis_label = "age",y_axis_label = "tenure")
fig.circle(df["age"],df["tenure"])
output_notebook()
show(fig)
fig.line(df["age"],df["tenure"])
show(fig)
fig.x(df["age"],df["expr"],line_color="red",size=10)
show(fig)
fig.line(df["age"],df["expr"],line_color="red")
show(fig)