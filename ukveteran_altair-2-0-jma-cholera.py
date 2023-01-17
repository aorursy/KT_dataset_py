import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import altair as alt

alt.renderers.enable('kaggle')

alt.data_transformers.enable('default', max_rows=None)
dat = pd.read_csv('../input/william-farrs-data-on-cholera-in-london/Cholera.csv')

dat.head()
line = alt.Chart(dat).mark_line().encode(

    x='cholera_drate',

    y='cholera_deaths'

)

line
alt.Chart(dat).mark_point().encode(

    x='cholera_drate',

    y='cholera_deaths'

)
base = alt.Chart(dat)



bar = base.mark_bar().encode(

    x=alt.X('cholera_deaths:Q', bin=True, axis=None),

    y='count()'

)



rule = base.mark_rule(color='red').encode(

    x='mean(cholera_deaths):Q',

    size=alt.value(5)

)



bar + rule
base = alt.Chart(dat)



bar = base.mark_bar().encode(

    x=alt.X('cholera_drate:Q', bin=True, axis=None),

    y='count()'

)



rule = base.mark_rule(color='red').encode(

    x='mean(cholera_drate):Q',

    size=alt.value(5)

)



bar + rule
alt.Chart(dat).mark_circle().encode(

    alt.X('cholera_deaths', scale=alt.Scale(zero=False)),

    alt.Y('cholera_drate', scale=alt.Scale(zero=False, padding=1)),

    color='elevation'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('cholera_deaths', scale=alt.Scale(zero=False)),

    alt.Y('cholera_drate', scale=alt.Scale(zero=False, padding=1)),

    color='region'

)