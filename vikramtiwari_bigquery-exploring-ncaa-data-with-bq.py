# offical google cloud library is recommended

from google.cloud import bigquery

import pandas as pd
# initiate bigquery client

bq = bigquery.Client()
# just write your sql query

sql_query = """

#standardSQL

SELECT

  event_type,

  COUNT(*) AS event_count

FROM `bigquery-public-data.ncaa_basketball.mbb_pbp_sr`

GROUP BY 1

ORDER BY event_count DESC;

"""

# and run the query to get the result in a dataframe

results = bq.query(sql_query).to_dataframe()

results
# just write your sql query

sql_query = """

#standardSQL

#most three points made

SELECT

  scheduled_date,

  name,

  market,

  alias,

  three_points_att,

  three_points_made,

  three_points_pct,

  opp_name,

  opp_market,

  opp_alias,

  opp_three_points_att,

  opp_three_points_made,

  opp_three_points_pct,

  (three_points_made + opp_three_points_made) AS total_threes

FROM `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr`

WHERE season > 2010

ORDER BY total_threes DESC

LIMIT 5;

"""

# and run the query to get the result in a dataframe

results = bq.query(sql_query).to_dataframe()

results
# just write your sql query

sql_query = """

#standardSQL

SELECT

  venue_name, venue_capacity, venue_city, venue_state

FROM `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr`

GROUP BY 1,2,3,4

ORDER BY venue_capacity DESC

LIMIT 5;

"""

# and run the query to get the result in a dataframe

results = bq.query(sql_query).to_dataframe()

results
# just write your sql query

sql_query = """

#standardSQL

#highest scoring game of all time

SELECT

  scheduled_date,

  name,

  market,

  alias,

  points_game AS team_points,

  opp_name,

  opp_market,

  opp_alias,

  opp_points_game AS opposing_team_points,

  points_game + opp_points_game AS point_total

FROM `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr`

WHERE season > 2010

ORDER BY point_total DESC

LIMIT 5;

"""

# and run the query to get the result in a dataframe

results = bq.query(sql_query).to_dataframe()

results
# just write your sql query

sql_query = """

#standardSQL

#biggest point difference in a championship game

SELECT

  scheduled_date,

  name,

  market,

  alias,

  points_game AS team_points,

  opp_name,

  opp_market,

  opp_alias,

  opp_points_game AS opposing_team_points,

  ABS(points_game - opp_points_game) AS point_difference

FROM `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr`

WHERE season > 2015 AND tournament_type = 'National Championship'

ORDER BY point_difference DESC

LIMIT 5;



"""

# and run the query to get the result in a dataframe

results = bq.query(sql_query).to_dataframe()

results