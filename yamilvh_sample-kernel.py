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
import pandas as pd

data = pd.read_csv("../input/data.csv")

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()
data.head()
dash = [go.Scatter(x=data.Agility, y=data.Balance, mode='markers', text= data.Name)]

layout = dict(title = "Agility and balance", hovermode= 'closest',

              xaxis= dict(title= 'Agility'))

fig = dict(data = dash, layout = layout)

iplot(fig)