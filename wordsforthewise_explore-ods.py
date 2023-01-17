from subprocess import check_output

import numpy as np

import pandas as pd

import re

from bokeh.io import output_notebook

from bokeh.sampledata import us_states

from bokeh.plotting import figure, show, output_file, ColumnDataSource

from bokeh.models import HoverTool, Range1d

import matplotlib.pyplot as plt

%matplotlib inline

output_notebook()



print(check_output(["ls", "../input"]).decode("utf8"))
basepath = '../input/'

opdf = pd.read_csv(basepath + 'opioids.csv')

oddf = pd.read_csv(basepath + 'overdoses.csv')

pdf = pd.read_csv(basepath + 'prescriber-info.csv')
opdf.head()
oddf.head()
pdf.head()
oddf['Deaths'] = oddf['Deaths'].apply(lambda x: float(re.sub(',', '', x)))

oddf['Population'] = oddf['Population'].apply(lambda x: float(re.sub(',', '', x)))
oddf = oddf.sort_values(by='Abbrev')

oddf = oddf[oddf['Abbrev'] != 'AK']

oddf = oddf[oddf['Abbrev'] != 'HI']
us_states = us_states.data.copy()

state_abbrevs = sorted(us_states.keys())

state_abbrevs.remove('DC')

state_abbrevs.remove('AK')

state_abbrevs.remove('HI')
state_xs = [us_states[code]["lons"] for code in state_abbrevs]

state_ys = [us_states[code]["lats"] for code in state_abbrevs]
min_x = min([min(s) for s in state_xs])

max_x = max([max(s) for s in state_xs])

min_y = min([min(s) for s in state_ys])

max_y = max([max(s) for s in state_ys])
# from here: http://nbviewer.jupyter.org/github/pybokeh/jupyter_notebooks/blob/master/bokeh/state_choropleth_example.ipynb

colors = ["#F1EEF6", "#D4B9DA", "#C994C7", "#DF65B0", "#DD1C77", "#980043"]

state_colors = []

maxDeaths = oddf['Deaths'].max()

for state in state_abbrevs:

    try:

        # get the value for the state

        rate = oddf[oddf['Abbrev']==state]['Deaths'].values[0]

        # Normalize the value by dividing it by the max value then multiply by the number of colors

        idx = int((rate/maxDeaths) * (len(colors) - 1) )

        state_colors.append(colors[idx])

    except KeyError:

        state_colors.append("black")

        

source = ColumnDataSource(oddf[['Abbrev', 'Deaths']])



TOOLS="pan,wheel_zoom,box_zoom,reset,hover,save,resize"



p = figure(title="ODs by State (absolute value)", toolbar_location="left",

           plot_width=600, plot_height=400, tools=TOOLS)



p.patches(state_xs, state_ys, source=source, fill_color=state_colors, fill_alpha=0.9,

          line_color="#884444", line_width=2, line_alpha=0.3)



hover = p.select_one(HoverTool)

hover.point_policy = "follow_mouse"

hover.tooltips = [

    ("State", "@Abbrev"),

    ("Deaths", "@Deaths"),

    ("(Long, Lat)", "($x, $y)")

]



p.x_range = Range1d(min_x, max_x)

p.y_range = Range1d(min_y, max_y)



show(p)
oddf['Relative_Deaths'] = oddf['Deaths'] / oddf['Population']
# from here: http://nbviewer.jupyter.org/github/pybokeh/jupyter_notebooks/blob/master/bokeh/state_choropleth_example.ipynb

colors = ["#F1EEF6", "#D4B9DA", "#C994C7", "#DF65B0", "#DD1C77", "#980043"]

state_colors = []

maxDeaths = oddf['Relative_Deaths'].max()

for state in state_abbrevs:

    try:

        # get the value for the state

        rate = oddf[oddf['Abbrev']==state]['Relative_Deaths'].values[0]

        # Normalize the value by dividing it by the max value then multiply by the number of colors

        idx = int((rate/maxDeaths) * (len(colors) - 1) )

        state_colors.append(colors[idx])

    except KeyError:

        state_colors.append("black")

        

source = ColumnDataSource(oddf[['Abbrev', 'Relative_Deaths']])



TOOLS="pan,wheel_zoom,box_zoom,reset,hover,save,resize"



p = figure(title="ODs by State (relative to population)", toolbar_location="left",

           plot_width=600, plot_height=400, tools=TOOLS)



p.patches(state_xs, state_ys, source=source, fill_color=state_colors, fill_alpha=0.9,

          line_color="#884444", line_width=2, line_alpha=0.3)



hover = p.select_one(HoverTool)

hover.point_policy = "follow_mouse"

hover.tooltips = [

    ("State", "@Abbrev"),

    ("Relative Deaths", "@Relative_Deaths"),

    ("(Long, Lat)", "($x, $y)")

]



p.x_range = Range1d(min_x, max_x)

p.y_range = Range1d(min_y, max_y)



show(p)
Blues = plt.cm.ScalarMappable(norm=[oddf['Deaths'].min(), oddf['Deaths'].max()], cmap='Blues')