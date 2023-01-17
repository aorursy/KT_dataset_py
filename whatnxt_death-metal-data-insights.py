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

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import builtins

import plotly.offline as py

py.init_notebook_mode(connected=True)

import matplotlib.pyplot as plt

df = pd.DataFrame()

bands = df.from_csv("../input/bands.csv",index_col=None)



bands_country = bands.groupby("country").size().to_frame()

bands_country.columns = ['count']

bands_country.reset_index(inplace=True)

data = [ dict(

        type = 'choropleth',

        locations = bands_country['country'],

        locationmode = "country names",

        z = bands_country['count'],

        text = bands_country['country'],

        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\

            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            tickprefix = '#',

            title = 'Number of Bands per country'),

      ) ]



layout = dict(

    title = 'No of bands per country',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    )

)



fig = dict( data=data, layout=layout )

py.iplot( fig, validate=False, filename='death-metal-band' )