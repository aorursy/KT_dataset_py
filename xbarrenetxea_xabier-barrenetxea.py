import sqlite3

import numpy as np

import pandas as pd

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler

from bokeh.plotting import figure, ColumnDataSource, show

from bokeh.models import HoverTool

from bokeh.io import output_notebook

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as NP

import math

import re



%matplotlib inline

output_notebook()



database = '../input/database.sqlite'

conn = sqlite3.connect(database)



# list all tables

query = "SELECT name as Tablas FROM sqlite_master WHERE type='table';"

pd.read_sql(query, conn)
# Siguiendo el paso 2 de CRISP-DM, efectuamos un conteo de las filas que contienen al menos un NULL

# en la base de datos. El número de filas por tabla viene dado en la descripción de la base de

# datos, por lo que no hay que calcularlo.

with sqlite3.connect('../input/database.sqlite') as con:

    player_attributes_null_amount = pd.read_sql_query("SELECT COUNT (*) from Player_Attributes WHERE id IS NULL OR player_fifa_api_id IS NULL OR player_api_id IS NULL OR date IS NULL OR overall_rating IS NULL OR potential IS NULL OR preferred_foot IS NULL OR attacking_work_rate IS NULL OR defensive_work_rate IS NULL OR crossing IS NULL OR finishing IS NULL OR heading_accuracy IS NULL OR short_passing IS NULL OR volleys IS NULL OR dribbling IS NULL OR curve IS NULL OR free_kick_accuracy IS NULL OR long_passing IS NULL OR ball_control IS NULL OR acceleration IS NULL OR sprint_speed IS NULL OR agility IS NULL OR reactions IS NULL OR balance IS NULL OR shot_power IS NULL OR jumping IS NULL OR stamina IS NULL OR strength IS NULL OR long_shots IS NULL OR aggression IS NULL OR interceptions IS NULL OR positioning IS NULL OR vision IS NULL OR penalties IS NULL OR marking IS NULL OR standing_tackle IS NULL OR sliding_tackle IS NULL OR gk_diving IS NULL OR gk_handling IS NULL OR gk_kicking IS NULL OR gk_positioning IS NULL OR gk_reflexes IS NULL", con)

    player_null_amount = pd.read_sql_query("SELECT COUNT (*) from Player WHERE id IS NULL OR player_api_id IS NULL OR player_name IS NULL OR player_fifa_api_id IS NULL OR birthday IS NULL OR height IS NULL OR weight IS NULL", con)

    match_null_amount = pd.read_sql_query("SELECT COUNT (*) from Match WHERE id IS NULL OR country_id IS NULL OR league_id IS NULL OR season IS NULL OR stage IS NULL OR date IS NULL OR match_api_id IS NULL OR home_team_api_id IS NULL OR away_team_api_id IS NULL OR home_team_goal IS NULL OR away_team_goal IS NULL OR home_player_X1 IS NULL OR home_player_X2 IS NULL OR home_player_X3 IS NULL OR home_player_X4 IS NULL OR home_player_X5 IS NULL OR home_player_X6 IS NULL OR home_player_X7 IS NULL OR home_player_X8 IS NULL OR home_player_X9 IS NULL OR home_player_X10 IS NULL OR home_player_X11 IS NULL OR away_player_X1 IS NULL OR away_player_X2 IS NULL OR away_player_X3 IS NULL OR away_player_X4 IS NULL OR away_player_X5 IS NULL OR away_player_X6 IS NULL OR away_player_X7 IS NULL OR away_player_X8 IS NULL OR away_player_X9 IS NULL OR away_player_X10 IS NULL OR away_player_X11 IS NULL OR home_player_Y1 IS NULL OR home_player_Y2 IS NULL OR home_player_Y3 IS NULL OR home_player_Y4 IS NULL OR home_player_Y5 IS NULL OR home_player_Y6 IS NULL OR home_player_Y7 IS NULL OR home_player_Y8 IS NULL OR home_player_Y9 IS NULL OR home_player_Y10 IS NULL OR home_player_Y11 IS NULL OR away_player_Y1 IS NULL OR away_player_Y2 IS NULL OR away_player_Y3 IS NULL OR away_player_Y4 IS NULL OR away_player_Y5 IS NULL OR away_player_Y6 IS NULL OR away_player_Y7 IS NULL OR away_player_Y8 IS NULL OR away_player_Y9 IS NULL OR away_player_Y10 IS NULL OR away_player_Y11 IS NULL OR home_player_1 IS NULL OR home_player_2 IS NULL OR home_player_3 IS NULL OR home_player_4 IS NULL OR home_player_5 IS NULL OR home_player_6 IS NULL OR home_player_7 IS NULL OR home_player_8 IS NULL OR home_player_9 IS NULL OR home_player_10 IS NULL OR home_player_11 IS NULL OR away_player_1 IS NULL OR away_player_2 IS NULL OR away_player_3 IS NULL OR away_player_4 IS NULL OR away_player_5 IS NULL OR away_player_6 IS NULL OR away_player_7 IS NULL OR away_player_8 IS NULL OR away_player_9 IS NULL OR away_player_10 IS NULL OR away_player_11 IS NULL OR goal IS NULL OR shoton IS NULL OR shotoff IS NULL OR foulcommit IS NULL OR card IS NULL OR cross IS NULL OR corner IS NULL OR possession IS NULL OR B365H IS NULL OR B365D IS NULL OR B365A IS NULL OR BWH IS NULL OR BWD IS NULL OR BWA IS NULL OR IWH IS NULL OR IWD IS NULL OR IWA IS NULL OR LBH IS NULL OR LBD IS NULL OR LBA IS NULL OR PSH IS NULL OR PSD IS NULL OR PSA IS NULL OR WHH IS NULL OR WHD IS NULL OR WHA IS NULL OR SJH IS NULL OR SJD IS NULL OR SJA IS NULL OR VCH IS NULL OR VCD IS NULL OR VCA IS NULL OR GBH IS NULL OR GBD IS NULL OR GBA IS NULL OR BSH IS NULL OR BSD IS NULL OR BSA IS NULL", con)

    league_null_amount = pd.read_sql_query("SELECT COUNT (*) from League WHERE id IS NULL OR country_id IS NULL OR name IS NULL", con)

    country_null_amount = pd.read_sql_query("SELECT COUNT (*) from Country WHERE id IS NULL or name IS NULL", con)

    team_null_amount = pd.read_sql_query("SELECT COUNT (*) from Team WHERE id IS NULL OR team_api_id IS NULL OR team_fifa_api_id IS NULL OR team_long_name IS NULL OR team_short_name IS NULL", con)

    team_attributes_null_amount = pd.read_sql_query("SELECT COUNT (*) from Team_Attributes WHERE id IS NULL OR team_fifa_api_id IS NULL OR team_api_id IS NULL OR date IS NULL OR buildUpPlaySpeed IS NULL OR buildUpPlaySpeedClass IS NULL OR buildUpPlayDribbling IS NULL OR buildUpPlayDribblingClass IS NULL OR buildUpPlayPassing IS NULL OR buildUpPlayPassingClass IS NULL OR buildUpPlayPositioningClass IS NULL OR chanceCreationPassing IS NULL OR chanceCreationPassingClass IS NULL OR chanceCreationCrossing IS NULL OR chanceCreationCrossingClass IS NULL OR chanceCreationShooting IS NULL OR chanceCreationShootingClass IS NULL OR chanceCreationPositioningClass IS NULL OR defencePressure IS NULL OR defencePressureClass IS NULL OR defenceAggression IS NULL OR defenceAggressionClass IS NULL OR defenceTeamWidth IS NULL OR defenceTeamWidthClass IS NULL OR defenceDefenderLineClass IS NULL", con)

con.close()

print ("Número de filas (de 183978) con algún NULL en la tabla Player_Attributes:")    

print (player_attributes_null_amount.head())

print ("*************************************************************************")

print ("Número de filas (de 11060) con algún NULL en la tabla Player:")    

print (player_null_amount.head())

print ("*************************************************************************")

print ("Número de filas (de 25979) con algún NULL en la tabla Match:")    

print (match_null_amount.head())

print ("*************************************************************************")

print ("Número de filas (de 11) con algún NULL en la tabla League:")    

print (league_null_amount.head())

print ("*************************************************************************")

print ("Número de filas (de 11) con algún NULL en la tabla Countries:")    

print (country_null_amount.head())

print ("*************************************************************************")

print ("Número de filas (de 299) con algún NULL en la tabla Team:")    

print (team_null_amount.head())

print ("*************************************************************************")

print ("Número de filas (de 1458) con algún NULL en la tabla Team_Attributes:")    

print (team_attributes_null_amount.head())

print ("*************************************************************************")
with sqlite3.connect('../input/database.sqlite') as con:

    player_attributes_all = pd.read_sql_query("SELECT * from Player_Attributes", con)

    player_all = pd.read_sql_query("SELECT * from Player", con)

    match_all = pd.read_sql_query("SELECT * from Match", con)

    league_all = pd.read_sql_query("SELECT * from League", con)

    country_all = pd.read_sql_query("SELECT * from Country", con)

    team_all = pd.read_sql_query("SELECT * from Team", con)

    team_attributes_all = pd.read_sql_query("SELECT * from Team_Attributes", con)

con.close()



print ("Aspecto del contenido de la tabla Player_Attributes:\n")

print (player_attributes_all.head())

print ("*************************************************************************")

print ("Aspecto del contenido de la tabla Player:\n")    

print (player_all.head())

print ("*************************************************************************")

print ("Aspecto del contenido de la tabla Match:\n")    

print (match_all.head())

print ("*************************************************************************")

print ("Aspecto del contenido de la tabla League:\n")    

print (league_all.head())

print ("*************************************************************************")

print ("Aspecto del contenido de la tabla Countries:\n")    

print (country_all.head())

print ("*************************************************************************")

print ("Aspecto del contenido de la tabla Team:\n")    

print (team_all.head())

print ("*************************************************************************")

print ("Aspecto del contenido de la tabla Team_Attributes:\n")    

print (team_attributes_all.head())

print ("*************************************************************************")
datos = pd.merge(player_attributes_all, player_all, on=['id', 'id'])

print('valores nulos para date',datos.date.isnull().sum())

print('valores nulos para birthday',datos.birthday.isnull().sum())

print('valores nulos para overall_rating',datos.overall_rating.isnull().sum())
datos=datos.dropna(subset = ['overall_rating'])

print('valores nulos para overall_rating',datos.overall_rating.isnull().sum())
#Generamos listas de los datos que vamos a requerir



date_year=[]

for row in datos.date:

    date_year.append(int(row[:4]))



birthday_year=[]

for row in datos.birthday:

    birthday_year.append(int(row[:4]))



player_name=[]

for row in datos.player_name:

    player_name.append(row)

        



overall_rating=[]

for row in datos.overall_rating:

    overall_rating.append(row)

        
#Calculamos la edad de los jugadores

a=zip(date_year,birthday_year)

age=[]

for row in a:

    age.append(row[0]-row[1])
#Generamos los datos necesarios agrupados para después poder pasar al punto 4

age_rating=[list(x) for x in zip(overall_rating,age)]

sortkeyfn = key=lambda s:s[1]

age_rating.sort(key=sortkeyfn)

from itertools import groupby

result = {}

for key,valuesiter in groupby(age_rating, key=sortkeyfn):

    result[key] = list(v[0] for v in valuesiter)

avgDict = {}

for k,v in result.items():



    avgDict[k] = sum(v)/ float(len(v))

    
#Por tanto tenemos lo siguiente

avgDict
plt.figure(figsize=(20, 10))



plt.bar(range(len(avgDict)), avgDict.values(), align='center',width=0.5)

plt.xticks(range(len(avgDict)), avgDict.keys())



plt.show()