import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 

iplot([{"x": [1, 2, 3], "y": [3, 1, 6]}])