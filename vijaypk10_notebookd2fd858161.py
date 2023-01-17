# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

iris = pd.read_csv("../input/Iris.csv")
iris.info()
from bokeh.layouts import row, column, gridplot, widgetbox

from bokeh.plotting import figure, ColumnDataSource

from bokeh.io import output_file, show, output_notebook, output_server, curdoc

from bokeh.models import HoverTool, CategoricalColorMapper, Slider, Select, Button

from bokeh.models.widgets import Panel, Tabs

from bokeh.charts import Histogram, BoxPlot, output_file, show, Scatter

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

iris = pd.read_csv("../input/Iris.csv")

p1 = figure(x_axis_label = "Sepal Length", y_axis_label = "Sepal Width")

p1.circle("SepalLengthCm", "SepalWidthCm", color="blue", source=iris)

p2 = figure(x_axis_label = "Petal Length", y_axis_label = "Petal Width")

p2.circle("PetalLengthCm", "PetalWidthCm", color="red", source = iris)

p3 = figure(x_axis_label = "Sepal Length", y_axis_label = "Petal length")

p3.circle("SepalLengthCm", "PetalLengthCm", color="green", source=iris)

output_notebook()

# layout = row(p1, p2, p3)

# output_file("iris_file")

show(p1)

# iris.shape()
from bokeh.io import curdoc, output_server

from bokeh.plotting import figure



plot=figure()

plot.line(x=[1,2,3,4,5],y=[2,5,4,6,9])

curdoc().add_root(plot)
import numpy as np
