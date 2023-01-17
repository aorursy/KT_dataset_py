import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import altair as alt

alt.renderers.enable('kaggle')

alt.data_transformers.enable('default', max_rows=None)
dat = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')

dat.head()
line = alt.Chart(dat).mark_line().encode(

    x='GRE Score',

    y='TOEFL Score'

)

line
alt.Chart(dat).mark_point().encode(

   x='GRE Score',

    y='TOEFL Score'

)
base = alt.Chart(dat)



bar = base.mark_bar().encode(

    x=alt.X('GRE Score:Q', bin=True, axis=None),

    y='count()'

)



rule = base.mark_rule(color='red').encode(

    x='mean(GRE Score):Q',

    size=alt.value(5)

)



bar + rule
base = alt.Chart(dat)



bar = base.mark_bar().encode(

    x=alt.X('TOEFL Score:Q', bin=True, axis=None),

    y='count()'

)



rule = base.mark_rule(color='red').encode(

    x='mean(TOEFL Score):Q',

    size=alt.value(5)

)



bar + rule
alt.Chart(dat).mark_circle().encode(

    alt.X('GRE Score', scale=alt.Scale(zero=False)),

    alt.Y('TOEFL Score', scale=alt.Scale(zero=False, padding=1)),

    color='Research'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('GRE Score', scale=alt.Scale(zero=False)),

    alt.Y('TOEFL Score', scale=alt.Scale(zero=False, padding=1)),

    color='University Rating'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('GRE Score', scale=alt.Scale(zero=False)),

    alt.Y('TOEFL Score', scale=alt.Scale(zero=False, padding=1)),

    size='University Rating'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('GRE Score', scale=alt.Scale(zero=False)),

    alt.Y('TOEFL Score', scale=alt.Scale(zero=False, padding=1)),

    color='Research',

    size='University Rating'

)
alt.Chart(dat).mark_circle().encode(

    alt.X(alt.repeat("column"), type='quantitative'),

    alt.Y(alt.repeat("row"), type='quantitative'),

    color='Origin:N'

).properties(

    width=150,

    height=150

).repeat(

    row=['GRE Score', 'TOEFL Score'],

    column=[ 'TOEFL Score', 'GRE Score']

).interactive()