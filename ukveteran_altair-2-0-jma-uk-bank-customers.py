import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import altair as alt

alt.renderers.enable('kaggle')

alt.data_transformers.enable('default', max_rows=None)
dat = pd.read_csv('../input/uk-bank-customers/P6-UK-Bank-Customers.csv')

dat.head()
line = alt.Chart(dat).mark_line().encode(

    x='Age',

    y='Balance'

)

line
alt.Chart(dat).mark_point().encode(

    x='Age',

    y='Balance'

)
base = alt.Chart(dat)



bar = base.mark_bar().encode(

    x=alt.X('Balance:Q', bin=True, axis=None),

    y='count()'

)



rule = base.mark_rule(color='red').encode(

    x='mean(Balance):Q',

    size=alt.value(5)

)



bar + rule
alt.Chart(dat).mark_circle().encode(

    alt.X('Age', scale=alt.Scale(zero=False)),

    alt.Y('Balance', scale=alt.Scale(zero=False, padding=1)),

    color='Gender'

)
alt.Chart(dat).mark_circle().encode(

    alt.X('Age', scale=alt.Scale(zero=False)),

    alt.Y('Balance', scale=alt.Scale(zero=False, padding=1)),

    color='Gender',

    size='Region'

)