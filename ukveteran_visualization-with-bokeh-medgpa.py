import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bokeh.io import output_file,show,output_notebook,save

from bokeh.plotting import figure
df = pd.read_csv("../input/gpa-and-medical-school-admission/MedGPA.csv")

df
fig = figure(x_axis_label = "GPA",y_axis_label = "MCAT")
fig.circle(df["GPA"],df["MCAT"])
output_notebook()
show(fig)
fig.line(df["GPA"],df["MCAT"])
show(fig)
fig.x(df["GPA"],df["MCAT"],line_color="red",size=10)
show(fig)
fig.line(df["GPA"],df["MCAT"],line_color="red")
show(fig)