import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



%matplotlib inline



data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

data.head()
data.info()
import plotly.graph_objects as go

fig = go.Figure(

    data=[go.Bar(y=[2, 1, 3])],

    layout_title_text="A Figure Displayed with fig.show()"

)

fig.show()