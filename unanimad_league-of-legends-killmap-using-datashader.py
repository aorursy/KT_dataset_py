import pandas as pd

import datashader as ds

from datashader import transfer_functions as tf

import numpy as np
df = pd.read_csv('/kaggle/input/leagueoflegends/kills.csv')
df.head()
df.info()
df = df.drop(['Address', 'Team', 'Victim', 'Killer', 'Assist_1', 'Assist_2', 'Assist_3', 'Assist_4'], axis=1)

df = df.rename(columns={'Time': 'time'})

df = df.loc[df.x_pos != 'TooEarly']

df = df.loc[df.x_pos.notnull()]

df.x_pos = df.x_pos.apply(lambda x: int(x))

df.y_pos = df.y_pos.apply(lambda y: int(y))
def create_phase_category(df):

    conditions = [

        (df.time<17),

        (17<df.time) & (df.time<32),

        (32<df.time)]



    choices = ['early', 'mid', 'late']

    df['phase'] = np.select(conditions, choices, default='very_early')

    df['phase'] = df['phase'].astype('category')

    

    return(df)
df.head()
df.describe()
kill_df = create_phase_category(df)
kill_df.head()
kill_df.describe()
kill_df.groupby(['phase'], as_index=False).count().sort_values(by='phase')
def visualise_with_datashader(df):

    color_key = {'very_early': 'black', 'early': 'lightyellow', 'mid': 'tomato', 'late': 'firebrick'}

    

    cvs = ds.Canvas()

    agg = cvs.points(df, 'x_pos', 'y_pos', ds.count_cat('phase'))

    

    img = tf.shade(agg, color_key=color_key, how='eq_hist')

    img = tf.set_background(img, 'black')

    

    return(img)
print(f'Total points : {kill_df.shape[0]}')



img = visualise_with_datashader(kill_df)
img
ds.utils.export_image(img=img, filename='league_of_legends_kill_map', fmt=".png");