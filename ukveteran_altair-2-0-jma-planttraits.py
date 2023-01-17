import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import altair as alt

alt.renderers.enable('kaggle')

alt.data_transformers.enable('default', max_rows=None)
dat = pd.read_csv('../input/plant-species-traits-data/plantTraits.csv')

dat.head()
line = alt.Chart(dat).mark_line().encode(

    x='height',

    y='begflow'

)

line
alt.Chart(dat).mark_point().encode(

 x='height',

    y='begflow'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('height', scale=alt.Scale(zero=False)),

    alt.Y('begflow', scale=alt.Scale(zero=False, padding=1)),

    color='mycor'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('height', scale=alt.Scale(zero=False)),

    alt.Y('begflow', scale=alt.Scale(zero=False, padding=1)),

    color='mycor',

    size='durflow'

)