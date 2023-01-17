# Load packages



import numpy as np

import pandas as pd

import os



# Load Queary Packages



from google.cloud import bigquery

client = bigquery.Client()

from bq_helper import BigQueryHelper

dataset_ref = client.dataset('ncaa_basketball', project='bigquery-public-data')

ncaa_dataset = client.get_dataset(dataset_ref)



# Load ploting packages



import matplotlib.pyplot as plt

import seaborn as sns

import plotnine as p9

plt.style.use('fivethirtyeight')



# import plotly

import chart_studio as plotly

plotly.tools.set_config_file(world_readable=True, sharing='public')

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.tools as tls

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as fig_fact

from plotly.subplots import make_subplots





%matplotlib inline
[x.table_id for x in client.list_tables(ncaa_dataset)]
ncaa_mbb_pbp_sr = client.get_table(ncaa_dataset.table('mbb_pbp_sr'))


query="""SELECT

  event_type,

  season,

  type,

  team_alias,

  team_market,

  team_name,

  team_basket,

  event_coord_x,

  event_coord_y,

  three_point_shot,

  shot_made

FROM `bigquery-public-data.ncaa_basketball.mbb_pbp_sr`

WHERE season > 2012 AND type = "fieldgoal"

ORDER BY season"""



# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

df_shots = query_job.to_dataframe()

df_shots
query="""SELECT

  #first 35 minutes of regulation

  COUNTIF(event_type = "threepointmade"

    AND elapsed_time_sec < 2100) AS threes_made_first35,

  COUNTIF((event_type = "threepointmade"

      OR event_type = "threepointmiss")

    AND elapsed_time_sec < 2100) AS threes_att_first35,

  COUNTIF(event_type = "threepointmade"

    AND elapsed_time_sec < 2100) / COUNTIF((event_type = "threepointmade"

      OR event_type = "threepointmiss")

    AND elapsed_time_sec < 2100) AS three_pt_pct_first35,

  #last five minutes of regulation

  COUNTIF(event_type = "threepointmade"

    AND elapsed_time_sec >= 2100) AS threes_made_last5,

  COUNTIF((event_type = "threepointmade"

      OR event_type = "threepointmiss")

    AND elapsed_time_sec >= 2100) AS threes_att_last5,

  COUNTIF(event_type = "threepointmade"

    AND elapsed_time_sec >= 2100) / COUNTIF((event_type = "threepointmade"

      OR event_type = "threepointmiss")

    AND elapsed_time_sec >= 2100) AS three_pt_pct_last5

FROM `bigquery-public-data.ncaa_basketball.mbb_pbp_sr`

WHERE home_division_alias = "D1"

  AND away_division_alias = "D1"

  """



# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

df_end_game = query_job.to_dataframe()

df_end_game
query="""SELECT

  season,

  count(CASE WHEN shot_made = True THEN 1 END) as total_made,

  count(type) as total_shots

FROM `bigquery-public-data.ncaa_basketball.mbb_pbp_sr`

WHERE season > 1990 AND type = "fieldgoal"

GROUP BY season

ORDER BY season"""



# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

df_shots_made = query_job.to_dataframe()

df_shots_made



# calculate the fieldgoal %



df_shots_made['acc'] = df_shots_made['total_made']/df_shots_made['total_shots']

df_shots_made

acc_plot = p9.ggplot()

acc_plot = acc_plot + p9.geom_point(df_shots_made, mapping=p9.aes( x = 'season', y = 'acc'))

acc_plot = acc_plot + p9.ggtitle('Shot Accuracy vs Season')

acc_plot
query="""SELECT

  season,

  three_point_shot,

  count(CASE WHEN shot_made = True THEN 1 END) as total_made,

  count(type) as total_shots

FROM `bigquery-public-data.ncaa_basketball.mbb_pbp_sr`

WHERE season > 1990 AND type = "fieldgoal"

GROUP BY season,

   three_point_shot

ORDER BY season, 

   three_point_shot"""



# Set up the query

query_job = client.query(query)

# API request - run the query, and return a pandas DataFrame

df_shots_made_three = query_job.to_dataframe()

df_shots_made_three



# calculate the fieldgoal %



df_shots_made_three['acc'] = df_shots_made_three['total_made']/df_shots_made_three['total_shots']

df_shots_made_three
acc_plot = p9.ggplot()

acc_plot = acc_plot + p9.geom_point(df_shots_made_three, mapping=p9.aes( x = 'season', y = 'acc',color = 'three_point_shot'))

acc_plot = acc_plot + p9.ggtitle('Shot Accuracy vs Season')

acc_plot
# update the x,y coordinates to half court



mask = (df_shots['team_basket'] == 'right')

z_valid = df_shots[mask]



df_shots['coord_x'] = df_shots['event_coord_x']

df_shots.loc[mask, 'coord_x'] = (94*12) - z_valid['event_coord_x'] 



df_shots['coord_y'] = df_shots['event_coord_y']

df_shots.loc[mask, 'coord_y'] = (50*12) - z_valid['event_coord_y'] 



df_shots
# convert and rotate 90 degrees counter clockwise

df_shots['coord_x'] = (df_shots['coord_x']/12)

df_shots['coord_y'] = 25 - (df_shots['coord_y']/12)

df_shots['x'] = -1*(df_shots['coord_y'])

df_shots['y'] = (df_shots['coord_x']) - 5.25
# create shot distance and shot angle



df_shots['shot_distance'] = np.floor(np.sqrt(np.multiply(df_shots['x'],df_shots['x']) + np.multiply(df_shots['y'],df_shots['y'])))

df_shots['shot_angle'] = np.arccos(np.divide(df_shots['x'],df_shots['shot_distance'])) * 180/np.pi
def circle_points(center = np.array([0,0]), radius = 1, npoints = 360):

    angles = np.linspace(start = 0, stop = 360, num = npoints,)

    xx = np.repeat(center[0],npoints) + radius * np.cos(angles * np.pi/180)

    yy = np.repeat(center[1],npoints) + radius*np.sin(angles * np.pi/180)

    data = {'x':xx,'y':yy}

    df = pd.DataFrame(data)

    return df
# Court plot



width = 50

height = 94 / 2

key_height = 19

inner_key_width = 12

outer_key_width = 16

backboard_width = 6

backboard_offset = 4

neck_length = 0.5

hoop_radius = 0.75

hoop_center_y = backboard_offset + neck_length + hoop_radius

three_point_radius = 23.75

three_point_side_radius = 22

three_point_side_height = 14

college_three_radius = 20.75



data = {'x':[width / 2, width / 2, -width / 2, -width / 2, width / 2] , 'y':[height, 0, 0, height, height]}

court_points = pd.DataFrame(data)

court_points['desc'] = np.repeat('perimeter',len(court_points.index))

court_points['y'] = court_points['y'] - 5.25



data = {'x':[outer_key_width / 2, outer_key_width / 2, -outer_key_width / 2, -outer_key_width / 2] , 'y':[0, key_height, key_height, 0]}

outer_key = pd.DataFrame(data)

outer_key['desc'] = np.repeat('outer_key',len(outer_key.index))

outer_key['y'] = outer_key['y'] - 5.25



#court_points = court_points.append(temp, ignore_index=True)



data = {'x':[-backboard_width / 2, backboard_width / 2] , 'y':[backboard_offset, backboard_offset]}

backboard = pd.DataFrame(data)

backboard['desc'] = np.repeat('backboard',len(backboard.index))

backboard['y'] = backboard['y'] - 5.25

#court_points = court_points.append(temp, ignore_index=True)



data = {'x':[0, 0] , 'y':[backboard_offset, backboard_offset + neck_length]}

neck = pd.DataFrame(data)

neck['desc'] = np.repeat('neck',len(neck.index))

neck['y'] = neck['y'] - 5.25

#court_points = court_points.append(temp, ignore_index=True)



foul_circle = circle_points(center = [0,key_height], radius = inner_key_width / 2)

foul_circle['desc'] = np.repeat('foul_circle_top',len(foul_circle.index))

foul_circle.loc[foul_circle['y'] < key_height,'desc'] = 'foul_circle_bottom'

foul_circle['y'] = foul_circle['y'] - 5.25



hoop = circle_points(center = [0,hoop_center_y], radius = hoop_radius)

hoop['desc'] = np.repeat('hoop',len(hoop.index))

hoop['y'] = hoop['y'] - 5.25



restricted = circle_points(center = [0,hoop_center_y], radius = 4)

restricted = restricted[restricted.y >= hoop_center_y]

restricted['desc'] = np.repeat('restricted',len(restricted.index))

restricted['y'] = restricted['y'] - 5.25



college_three_circle = circle_points(center = [0,hoop_center_y], radius = college_three_radius)

college_three_circle = college_three_circle[college_three_circle.y >= hoop_center_y]

college_three_circle['y'] = college_three_circle['y'] - 5.25



data = {'x':[ -college_three_radius, -college_three_radius] , 'y':[ hoop_center_y, 0]}

college_three_line = pd.DataFrame(data)

college_three_line['y'] = college_three_line['y'] - 5.25



data = {'x':[ college_three_radius, college_three_radius] , 'y':[ hoop_center_y, 0]}

college_three_linea = pd.DataFrame(data)

college_three_linea['y'] = college_three_linea['y'] - 5.25





college_key = pd.DataFrame()

data = {'x':[inner_key_width / 2, inner_key_width / 2, -inner_key_width / 2, -inner_key_width / 2] , 'y':[0, key_height, key_height, 0]}

temp = pd.DataFrame(data)

temp['desc'] = np.repeat('college_key',len(temp.index))



college_key = college_key.append(temp, ignore_index=True)

college_key['y'] = college_key['y'] - 5.25
college_court = p9.ggplot()

college_court = college_court + p9.geom_path(court_points, mapping=p9.aes(x='x',y='y'))

college_court = college_court + p9.geom_path(outer_key, mapping=p9.aes(x='x',y='y'))

college_court = college_court + p9.geom_path(backboard, mapping=p9.aes(x='x',y='y'))

college_court = college_court + p9.geom_path(foul_circle, mapping=p9.aes(x='x',y='y'))

college_court = college_court + p9.geom_path(hoop, mapping=p9.aes(x='x',y='y'))

college_court = college_court + p9.geom_path(restricted, mapping=p9.aes(x='x',y='y'))

college_court = college_court + p9.geom_path(college_three_circle, mapping=p9.aes(x='x',y='y'))

college_court = college_court + p9.geom_path(college_key, mapping=p9.aes(x='x',y='y'))

college_court = college_court + p9.geom_path(college_three_line, mapping=p9.aes(x='x',y='y'))

college_court = college_court + p9.geom_path(college_three_linea, mapping=p9.aes(x='x',y='y'))



college_court
# show some team shot maps add points



df_plot_shot = df_shots[(df_shots['team_alias'] == 'UVA') | (df_shots['team_alias'] == 'MSU') |(df_shots['team_alias'] == 'MICH')|(df_shots['team_alias'] == 'DUKE')|(df_shots['team_alias'] == 'KU')]

df_plot_shot = df_plot_shot[df_plot_shot['y'] < 50 ]



shot_map = college_court + p9.geom_point(df_plot_shot, mapping=p9.aes( x= 'x', y= 'y', color = 'shot_made'),alpha =0.5)

shot_map = shot_map + p9.theme_bw(12) + p9.facet_grid('team_alias ~ season') + p9.ggtitle('Shot Maps') + p9.theme(aspect_ratio=1,figure_size=(12, 12))

shot_map
avg_shot_distance = np.mean(df_shots['shot_distance'])

avg_shot_distance
df_fg = df_shots[df_shots.type.isin(['fieldgoal'])& (df_shots['shot_distance'] < 50)].groupby(["shot_distance"]).agg({"shot_made":"count"}).rename(columns={"shot_made":"fga"})

df_fgm = df_shots[df_shots.type.isin(['fieldgoal'])& (df_shots['shot_distance'] < 50) & (df_shots['shot_made'] == True)].groupby(["shot_distance"]).agg({"shot_made":"count"}).rename(columns={"shot_made":"fgm"})

df_fg_dist = pd.merge(left=df_fg,right=df_fgm, left_on='shot_distance', right_on='shot_distance')

#df_fg_dist['distance'] = np.arange(0,50)

df_fg_dist.reset_index(inplace=True)



df_fg_dist['acc'] = df_fg_dist['fgm']/df_fg_dist['fga']

df_fg_dist



acc_plot = p9.ggplot()

acc_plot = acc_plot + p9.geom_point(df_fg_dist, mapping=p9.aes( x = 'shot_distance', y = 'acc'))

acc_plot = acc_plot + p9.ggtitle('Shot Accuracy vs Shot Distance')

acc_plot
df_fg = df_shots[df_shots.type.isin(['fieldgoal'])& (df_shots['shot_distance'] < 30)].groupby(["shot_distance","season"]).agg({"shot_made":"count"}).rename(columns={"shot_made":"fga"})

df_fgm = df_shots[df_shots.type.isin(['fieldgoal'])& (df_shots['shot_distance'] < 30) & (df_shots['shot_made'] == True)].groupby(["shot_distance","season"]).agg({"shot_made":"count"}).rename(columns={"shot_made":"fgm"})

df_fg_dist = pd.merge(left=df_fg,right=df_fgm, how = 'left', left_on=['shot_distance','season'], right_on=['shot_distance','season'])

df_fg_dist.reset_index(inplace=True)



df_fg_dist['acc'] = df_fg_dist['fgm']/df_fg_dist['fga']



acc_plot = p9.ggplot()

acc_plot = acc_plot + p9.geom_point(df_fg_dist, mapping=p9.aes( x = 'shot_distance', y = 'acc'))

acc_plot = acc_plot +  p9.facet_grid('. ~ season') + p9.ggtitle('Shot Accuracy vs Shot Distance By Year') + p9.theme(figure_size=(24, 8))

acc_plot