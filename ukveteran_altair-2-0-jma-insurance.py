import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import altair as alt

alt.renderers.enable('kaggle')

alt.data_transformers.enable('default', max_rows=None)
dat = pd.read_csv('../input/insurance/insurance.csv')

dat.head()
line = alt.Chart(dat).mark_line().encode(

    x='bmi',

    y='charges'

)

line
alt.Chart(dat).mark_point().encode(

    x='bmi',

    y='charges'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('bmi', scale=alt.Scale(zero=False)),

    alt.Y('charges', scale=alt.Scale(zero=False, padding=1)),

    color='sex'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('bmi', scale=alt.Scale(zero=False)),

    alt.Y('charges', scale=alt.Scale(zero=False, padding=1)),

    color='smoker'

)

alt.Chart(dat).mark_circle().encode(

    alt.X('bmi', scale=alt.Scale(zero=False)),

    alt.Y('charges', scale=alt.Scale(zero=False, padding=1)),

    color='sex',

    size='smoker'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('bmi', scale=alt.Scale(zero=False)),

    alt.Y('charges', scale=alt.Scale(zero=False, padding=1)),

    color='sex',

    size='region'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('bmi', scale=alt.Scale(zero=False)),

    alt.Y('charges', scale=alt.Scale(zero=False, padding=1)),

    color='smoker',

    size='region'

)