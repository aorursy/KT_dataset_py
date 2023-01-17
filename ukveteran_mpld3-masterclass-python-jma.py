import jinja2

import json

import numpy as np

import matplotlib.pyplot as plt



import mpld3

from mpld3 import plugins, utils





class HighlightLines(plugins.PluginBase):

    """A plugin to highlight lines on hover"""



    JAVASCRIPT = """

    mpld3.register_plugin("linehighlight", LineHighlightPlugin);

    LineHighlightPlugin.prototype = Object.create(mpld3.Plugin.prototype);

    LineHighlightPlugin.prototype.constructor = LineHighlightPlugin;

    LineHighlightPlugin.prototype.requiredProps = ["line_ids"];

    LineHighlightPlugin.prototype.defaultProps = {alpha_bg:0.3, alpha_fg:1.0}

    function LineHighlightPlugin(fig, props){

        mpld3.Plugin.call(this, fig, props);

    };



    LineHighlightPlugin.prototype.draw = function(){

      for(var i=0; i<this.props.line_ids.length; i++){

         var obj = mpld3.get_element(this.props.line_ids[i], this.fig),

             alpha_fg = this.props.alpha_fg;

             alpha_bg = this.props.alpha_bg;

         obj.elements()

             .on("mouseover", function(d, i){

                            d3.select(this).transition().duration(50)

                              .style("stroke-opacity", alpha_fg); })

             .on("mouseout", function(d, i){

                            d3.select(this).transition().duration(200)

                              .style("stroke-opacity", alpha_bg); });

      }

    };

    """



    def __init__(self, lines):

        self.lines = lines

        self.dict_ = {"type": "linehighlight",

                      "line_ids": [utils.get_id(line) for line in lines],

                      "alpha_bg": lines[0].get_alpha(),

                      "alpha_fg": 1.0}





N_paths = 50

N_steps = 100



x = np.linspace(0, 10, 100)

y = 0.1 * (np.random.random((N_paths, N_steps)) - 0.5)

y = y.cumsum(1)



fig, ax = plt.subplots(subplot_kw={'xticks': [], 'yticks': []})

lines = ax.plot(x, y.T, color='blue', lw=4, alpha=0.1)

plugins.connect(fig, HighlightLines(lines))



mpld3.display()
import matplotlib.pyplot as plt

import jinja2

import json

import numpy as np

import pandas as pd

import mpld3

from mpld3 import plugins

np.random.seed(9615)



# generate df

N = 100

df = pd.DataFrame((.1 * (np.random.random((N, 5)) - .5)).cumsum(0),

                  columns=['a', 'b', 'c', 'd', 'e'],)



# plot line + confidence interval

fig, ax = plt.subplots()

ax.grid(True, alpha=0.3)



for key, val in df.iteritems():

    l, = ax.plot(val.index, val.values, label=key)

    ax.fill_between(val.index,

                    val.values * .5, val.values * 1.5,

                    color=l.get_color(), alpha=.4)



# define interactive legend



handles, labels = ax.get_legend_handles_labels() # return lines and labels

interactive_legend = plugins.InteractiveLegendPlugin(zip(handles,

                                                         ax.collections),

                                                     labels,

                                                     alpha_unsel=0.5,

                                                     alpha_over=1.5, 

                                                     start_visible=True)

plugins.connect(fig, interactive_legend)



ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_title('Interactive legend', size=20)



mpld3.show()
%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np
# Scatter points

fig, ax = plt.subplots()

np.random.seed(0)

x, y = np.random.normal(size=(2, 200))

color, size = np.random.random((2, 200))



ax.scatter(x, y, c=color, s=500 * size, alpha=0.3)

ax.grid(color='lightgray', alpha=0.7)
# Draw lines

fig, ax = plt.subplots()

x = np.linspace(-5, 15, 1000)

for offset in np.linspace(0, 3, 4):

    ax.plot(x, 0.9 * np.sin(x - offset), lw=5, alpha=0.4,

            label="Offset: {0}".format(offset))

ax.set_xlim(0, 10)

ax.set_ylim(-1.2, 1.0)

ax.text(5, -1.1, "Here are some curves", size=18, ha='center')

ax.grid(color='lightgray', alpha=0.7)

ax.legend()
# multiple subplots, shared axes

fig, ax = plt.subplots(2, 2, figsize=(8, 6),sharex='col', sharey='row')

fig.subplots_adjust(hspace=0.3)



np.random.seed(0)



for axi in ax.flat:

    color = np.random.random(3)

    axi.plot(np.random.random(30), lw=2, c=color)

    axi.set_title("RGB = ({0:.2f}, {1:.2f}, {2:.2f})".format(*color),

                  size=14)

    axi.grid(color='lightgray', alpha=0.7)
from mpld3 import plugins



fig, ax = plt.subplots(3, 3, figsize=(6, 6))

fig.subplots_adjust(hspace=0.1, wspace=0.1)

ax = ax[::-1]



X = np.random.normal(size=(3, 100))

for i in range(3):

    for j in range(3):

        ax[i, j].xaxis.set_major_formatter(plt.NullFormatter())

        ax[i, j].yaxis.set_major_formatter(plt.NullFormatter())

        points = ax[i, j].scatter(X[j], X[i])

        

plugins.connect(fig, plugins.LinkedBrush(points))
N_paths = 5

N_steps = 100



x = np.linspace(0, 10, 100)

y = 0.1 * (np.random.random((N_paths, N_steps)) - 0.5)

y = y.cumsum(1)





fig = plt.figure()

ax1 = fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,2)



labels = ["a", "b", "c", "d", "e"]

l1 = ax1.plot(x, y.T, marker='x',lw=2, alpha=0.1)

s1 = ax2.plot(x, y.T, 'o', ms=8, alpha=0.1)

    

plugins.connect(fig, plugins.InteractiveLegendPlugin(zip(l1, s1), labels))

mpld3.display()
np.random.seed(0)

plt.plot(np.random.rand(10));
from mpld3 import plugins



fig, ax = plt.subplots()

ax.plot(np.random.random(10))

plugins.clear(fig)  # clear all plugins from the figure
from mpld3 import plugins



fig, ax = plt.subplots()

ax.plot(np.random.random(10))

plugins.clear(fig)  # clear all plugins from the figure



plugins.connect(fig, plugins.Reset(), plugins.BoxZoom(), plugins.Zoom())
fig, ax = plt.subplots()

points = ax.scatter(np.random.rand(40), np.random.rand(40),

                    s=300, alpha=0.3)



labels = ["Point {0}".format(i) for i in range(40)]

tooltip = plugins.PointLabelTooltip(points, labels)



plugins.connect(fig, tooltip)
from mpld3 import utils



class ClickInfo(plugins.PluginBase):

    """Plugin for getting info on click"""

    

    JAVASCRIPT = """

    mpld3.register_plugin("clickinfo", ClickInfo);

    ClickInfo.prototype = Object.create(mpld3.Plugin.prototype);

    ClickInfo.prototype.constructor = ClickInfo;

    ClickInfo.prototype.requiredProps = ["id"];

    function ClickInfo(fig, props){

        mpld3.Plugin.call(this, fig, props);

    };

    

    ClickInfo.prototype.draw = function(){

        var obj = mpld3.get_element(this.props.id);

        obj.elements().on("mousedown",

                          function(d, i){alert("clicked on points[" + i + "]");});

    }

    """

    def __init__(self, points):

        self.dict_ = {"type": "clickinfo",

                      "id": utils.get_id(points)}

        

fig, ax = plt.subplots()

points = ax.scatter(np.random.rand(50), np.random.rand(50),

                    s=500, alpha=0.3)



plugins.connect(fig, ClickInfo(points))
import numpy as np

import json

import matplotlib

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris



import mpld3

from mpld3 import plugins, utils





data = load_iris()

X = data.data

y = data.target



# dither the data for clearer plotting

X += 0.1 * np.random.random(X.shape)



fig, ax = plt.subplots(4, 4, sharex="col", sharey="row", figsize=(8, 8))

fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,

                    hspace=0.1, wspace=0.1)



for i in range(4):

    for j in range(4):

        points = ax[3 - i, j].scatter(X[:, j], X[:, i],

                                      c=y, s=40, alpha=0.6)



# remove tick labels

for axi in ax.flat:

    for axis in [axi.xaxis, axi.yaxis]:

        axis.set_major_formatter(plt.NullFormatter())



# Here we connect the linked brush plugin

plugins.connect(fig, plugins.LinkedBrush(points))



mpld3.show()