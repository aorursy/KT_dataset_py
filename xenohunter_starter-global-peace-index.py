import numpy as np

import pandas as pd



import altair as alt

alt.themes.enable('default')

alt.renderers.enable('kaggle');
START_YEAR = 2008

END_YEAR = 2019

YEARS = range(START_YEAR, END_YEAR + 1)
gpi = pd.read_csv(f'../input/global-peace-index/gpi-{START_YEAR}-{END_YEAR}.csv')
gpi.head()
gpi.tail()
score_columns = [f'{year} score' for year in YEARS]



total_score = pd.DataFrame({

    'year': list(map(str, YEARS)),

    'hostility': gpi[score_columns].sum()

})



alt.Chart(total_score).mark_line().encode(

    x='year',

    y=alt.Y('hostility', scale=alt.Scale(zero=False))

)