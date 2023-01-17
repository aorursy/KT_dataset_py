import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bokeh.io import output_file,show,output_notebook,save

from bokeh.plotting import figure
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

df
fig = figure(x_axis_label = "Glucose",y_axis_label = "Insulin")
fig.circle(df["Glucose"],df["Insulin"])
output_notebook()
show(fig)
fig.line(df["Glucose"],df["Insulin"])
show(fig)
fig.x(df["Glucose"],df["Insulin"],line_color="red",size=10)
show(fig)
fig = figure(x_axis_label = "Age",y_axis_label = "BMI")
fig.circle(df["Age"],df["BMI"])
output_notebook()
show(fig)
fig.line(df["Age"],df["BMI"])
show(fig)
fig.x(df["Age"],df["BMI"],line_color="red",size=10)
show(fig)