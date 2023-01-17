# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")

df.head()
data = df.iloc[:, 1:-1]

data.describe()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

stats = pd.DataFrame(scaler.fit_transform(data)).describe()

stats
stats.columns = data.columns

stats
from math import pi

from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter

from bokeh.plotting import figure

from bokeh.palettes import viridis

from bokeh.layouts import Column

from bokeh.io import output_notebook, show



def plot_vbar(index, x_range_link=None):



    # setup data source

    source = ColumnDataSource(data=dict(

        x=stats.columns.values,

        top=stats.loc[index,:].values,

        color=viridis(256)

    ))

    

    # plotting

    p = figure(title=index, plot_width=3000, plot_height=600, x_range=stats.columns.values if x_range_link is None else x_range_link)

    p.vbar(x="x", top="top", bottom=0, width=0.5, color="color", source=source)

    

    # add hover tool

    hover_tool = HoverTool(

        tooltips=[

            ("column", "@x"),

            (index, "@top{0,0.000000000000000000}")

        ],

        mode="vline"

    )

    p.add_tools(hover_tool)

    

    # adjust format

    p.xaxis[0].major_label_orientation = pi/4

    p.yaxis[0].formatter = NumeralTickFormatter(format="0,0.000000000000000000")

    

    return p



mean = plot_vbar(index="mean")

std = plot_vbar(index="std", x_range_link=mean.x_range)

min = plot_vbar(index="min", x_range_link=mean.x_range)

q25 = plot_vbar(index="25%", x_range_link=mean.x_range)

q50 = plot_vbar(index="50%", x_range_link=mean.x_range)

q75 = plot_vbar(index="75%", x_range_link=mean.x_range)

max = plot_vbar(index="max", x_range_link=mean.x_range)



output_notebook()

show(Column(mean, std, min, q25, q50, q75, max))