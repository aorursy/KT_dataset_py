from bokeh.io import output_file,show,output_notebook,push_notebook

from bokeh.plotting import figure

from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper

from bokeh.layouts import row,column,gridplot

from bokeh.models.widgets import Tabs,Panel

output_notebook()
from bokeh.plotting import figure, output_file, show



# prepare some data

x = [1, 2, 3, 4, 5]

y = [6, 7, 2, 4, 5]



# output to static HTML file

output_file("lines.html")



# create a new plot with a title and axis labels

p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')



# add a line renderer with legend and line thickness

p.line(x, y, legend="Temp.", line_width=2)



# show the results

show(p)
from bokeh.plotting import figure, output_file, show



# prepare some data

x = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

y0 = [i**2 for i in x]

y1 = [10**i for i in x]

y2 = [10**(i**2) for i in x]



# output to static HTML file

output_file("log_lines.html")



# create a new plot

p = figure(

   tools="pan,box_zoom,reset,save",

   y_axis_type="log", y_range=[0.001, 10**11], title="log axis example",

   x_axis_label='sections', y_axis_label='particles'

)



# add some renderers

p.line(x, x, legend="y=x")

p.circle(x, x, legend="y=x", fill_color="white", size=8)

p.line(x, y0, legend="y=x^2", line_width=3)

p.line(x, y1, legend="y=10^x", line_color="red")

p.circle(x, y1, legend="y=10^x", fill_color="red", line_color="red", size=6)

p.line(x, y2, legend="y=10^x^2", line_color="orange", line_dash="4 4")



# show the results

show(p)
import numpy as np



from bokeh.layouts import gridplot

from bokeh.plotting import figure, output_file, show



# prepare some data

N = 100

x = np.linspace(0, 4*np.pi, N)

y0 = np.sin(x)

y1 = np.cos(x)

y2 = np.sin(x) + np.cos(x)



# output to static HTML file

output_file("linked_panning.html")



# create a new plot

s1 = figure(width=250, plot_height=250, title=None)

s1.circle(x, y0, size=10, color="navy", alpha=0.5)



# NEW: create a new plot and share both ranges

s2 = figure(width=250, height=250, x_range=s1.x_range, y_range=s1.y_range, title=None)

s2.triangle(x, y1, size=10, color="firebrick", alpha=0.5)



# NEW: create a new plot and share only one range

s3 = figure(width=250, height=250, x_range=s1.x_range, title=None)

s3.square(x, y2, size=10, color="olive", alpha=0.5)



# NEW: put the subplots in a gridplot

p = gridplot([[s1, s2, s3]], toolbar_location=None)



# show the results

show(p)
from numpy import pi, arange, sin, linspace



from bokeh.plotting import output_file, figure, show

from bokeh.models import LinearAxis, Range1d



x = arange(-2*pi, 2*pi, 0.1)

y = sin(x)

y2 = linspace(0, 100, len(y))



output_file("twin_axis.html")



p = figure(x_range=(-6.5, 6.5), y_range=(-1.1, 1.1))



p.circle(x, y, color="red")



p.extra_y_ranges = {"foo": Range1d(start=0, end=100)}

p.circle(x, y2, color="blue", y_range_name="foo")

p.add_layout(LinearAxis(y_range_name="foo"), 'left')



show(p)
import numpy as np

from bokeh.plotting import figure, show

from bokeh.transform import linear_cmap

from bokeh.util.hex import hexbin



n = 50000

x = np.random.standard_normal(n)

y = np.random.standard_normal(n)



bins = hexbin(x, y, 0.1)



p = figure(tools="", match_aspect=True, background_fill_color='#440154')

p.grid.visible = False



p.hex_tile(q="q", r="r", size=0.1, line_color=None, source=bins,

           fill_color=linear_cmap('counts', 'Viridis256', 0, max(bins.counts)))



show(p)