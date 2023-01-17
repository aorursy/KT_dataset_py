import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import altair as alt

alt.renderers.enable('kaggle')

alt.data_transformers.enable('default', max_rows=None)
dat = pd.read_csv('../input/gpa-and-medical-school-admission/MedGPA.csv')

dat.head()
line = alt.Chart(dat).mark_line().encode(

    x='GPA',

    y='MCAT'

)

line
alt.Chart(dat).mark_point().encode(

    x='GPA',

    y='MCAT'

)
base = alt.Chart(dat)



bar = base.mark_bar().encode(

    x=alt.X('GPA:Q', bin=True, axis=None),

    y='count()'

)



rule = base.mark_rule(color='red').encode(

    x='mean(GPA):Q',

    size=alt.value(5)

)



bar + rule
base = alt.Chart(dat)



bar = base.mark_bar().encode(

    x=alt.X('MCAT:Q', bin=True, axis=None),

    y='count()'

)



rule = base.mark_rule(color='red').encode(

    x='mean(MCAT):Q',

    size=alt.value(5)

)



bar + rule
alt.Chart(dat).mark_circle().encode(

    alt.X('GPA', scale=alt.Scale(zero=False)),

    alt.Y('MCAT', scale=alt.Scale(zero=False, padding=1)),

    color='Sex'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('GPA', scale=alt.Scale(zero=False)),

    alt.Y('MCAT', scale=alt.Scale(zero=False, padding=1)),

    color='Accept'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('GPA', scale=alt.Scale(zero=False)),

    alt.Y('MCAT', scale=alt.Scale(zero=False, padding=1)),

    color='Acceptance'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('GPA', scale=alt.Scale(zero=False)),

    alt.Y('MCAT', scale=alt.Scale(zero=False, padding=1)),

    color='Sex',

    size='Accept'

)
alt.Chart(dat).mark_circle().encode(

    alt.X(alt.repeat("column"), type='quantitative'),

    alt.Y(alt.repeat("row"), type='quantitative'),

    color='Origin:N'

).properties(

    width=150,

    height=150

).repeat(

    row=['GPA', 'MCAT', 'Apps'],

    column=['Apps', 'MCAT', 'GPA']

).interactive()