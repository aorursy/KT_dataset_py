import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import altair as alt

alt.renderers.enable('kaggle')

alt.data_transformers.enable('default', max_rows=None)
dat = pd.read_csv('../input/malefemale-admissions/MaleFemaleAd.csv')

dat.head()
line = alt.Chart(dat).mark_line().encode(

    x='Total_Admissions',

    y='Admissions_Males'

)

line
line = alt.Chart(dat).mark_line().encode(

    x='Total_Admissions',

    y='Admissions_Females'

)

line
alt.Chart(dat).mark_circle().encode(

    alt.X('Admissions_Males', scale=alt.Scale(zero=False)),

    alt.Y('Admissions_Females', scale=alt.Scale(zero=False, padding=1)),

    color='CCGname'

)