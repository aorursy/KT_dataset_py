import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import altair as alt

alt.renderers.enable('kaggle')

alt.data_transformers.enable('default', max_rows=None)
dat = pd.read_csv('../input/postcoma-recovery-of-iq/Wong.csv')

dat.head()
line = alt.Chart(dat).mark_line().encode(

    x='duration',

    y='age'

)

line
alt.Chart(dat).mark_point().encode(

    x='duration',

    y='age'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('duration', scale=alt.Scale(zero=False)),

    alt.Y('age', scale=alt.Scale(zero=False, padding=1)),

    color='sex'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('duration', scale=alt.Scale(zero=False)),

    alt.Y('age', scale=alt.Scale(zero=False, padding=1)),

    color='days'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('duration', scale=alt.Scale(zero=False)),

    alt.Y('age', scale=alt.Scale(zero=False, padding=1)),

    color='sex',

    size='days'

)
alt.Chart(dat).mark_circle().encode(

    alt.X(alt.repeat("column"), type='quantitative'),

    alt.Y(alt.repeat("row"), type='quantitative'),

    color='Origin:N'

).properties(

    width=150,

    height=150

).repeat(

    row=['duration', 'age','piq', 'viq'],

    column=[ 'viq','piq','age', 'duration']

).interactive()