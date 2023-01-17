import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns



import altair as alt

alt.data_transformers.enable('default', max_rows=None)

dat = pd.read_csv('../input/icu-patients/ICU.csv')

dat.head()
line = alt.Chart(dat).mark_line().encode(

    x='Age',

    y='Pulse'

)

line
alt.Chart(dat).mark_point().encode(

    x='Age',

    y='Pulse'

)
alt.Chart(dat).mark_circle().encode(

    alt.X(alt.repeat("column"), type='quantitative'),

    alt.Y(alt.repeat("row"), type='quantitative'),

    color='Origin:N'

).properties(

    width=150,

    height=150

).repeat(

    row=['Age', 'SysBP', 'Pulse'],

    column=['Pulse', 'SysBP', 'Age']

).interactive()
base = alt.Chart(dat)



bar = base.mark_bar().encode(

    x=alt.X('Age:Q', bin=True, axis=None),

    y='count()'

)



rule = base.mark_rule(color='red').encode(

    x='mean(Age):Q',

    size=alt.value(5)

)



bar + rule
base = alt.Chart(dat)



bar = base.mark_bar().encode(

    x=alt.X('Pulse:Q', bin=True, axis=None),

    y='count()'

)



rule = base.mark_rule(color='red').encode(

    x='mean(Pulse):Q',

    size=alt.value(5)

)



bar + rule
alt.Chart(dat).mark_rect().encode(

    alt.X('Age:Q', bin=alt.Bin(maxbins=60)),

    alt.Y('Pulse:Q', bin=alt.Bin(maxbins=40)),

    alt.Color('count(Age):Q', scale=alt.Scale(scheme='greenblue'))

)
alt.Chart(dat).mark_point().encode(

    x='Age:Q',

    y='Pulse:Q'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('Age', scale=alt.Scale(zero=False)),

    alt.Y('Pulse', scale=alt.Scale(zero=False, padding=1)),

    color='Sex'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('Age', scale=alt.Scale(zero=False)),

    alt.Y('Pulse', scale=alt.Scale(zero=False, padding=1)),

    color='Emergency'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('Age', scale=alt.Scale(zero=False)),

    alt.Y('Pulse', scale=alt.Scale(zero=False, padding=1)),

    color='AgeGroup'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('Age', scale=alt.Scale(zero=False)),

    alt.Y('Pulse', scale=alt.Scale(zero=False, padding=1)),

    color='Sex',

    size= 'Emergency'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('Age', scale=alt.Scale(zero=False)),

    alt.Y('Pulse', scale=alt.Scale(zero=False, padding=1)),

    color='Sex',

    size= 'AgeGroup'

)