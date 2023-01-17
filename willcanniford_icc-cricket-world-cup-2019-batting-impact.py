# Imports for loading and cleaning data

import pandas as pd

import re

import numpy as np



# Imports for visualisations

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns



# Imports for batsman grouping and classification

from sklearn.cluster import KMeans

from sklearn import preprocessing 
world_cup_batting_raw = pd.read_csv("../input/world_cup_batting_raw.csv")

df = world_cup_batting_raw.loc[:, ['Player','Runs', 'BF','SR','4s','6s']].copy()

df.head()
def extract_country(player_string):

    regex = re.compile(r'.* \(([A-Z]*)\)')

    return(regex.search(player_string).group(1))



def clean_player_name(player_string):

    regex = re.compile(r'([a-zA-Z \-]*)\s\([A-Z]*\)')

    return(regex.search(player_string).group(1))
df['Country'] = df.Player.apply(extract_country) # Create separate `Country` column

df['Player'] = df.Player.apply(clean_player_name) # Clean and replace `Player`
df.head() # Inspect new format 
df['BoundaryRuns'] = df['4s'] * 4 + df['6s'] * 6

df['NonBoundaryRuns'] = df['Runs'] - df['BoundaryRuns']

df['TotalBoundaries'] = df['4s'] + df['6s']

df['NonBoundaryBalls'] = df['BF'] - df['TotalBoundaries']

df['RunsFromBoundary %'] = round(df['BoundaryRuns'] / df['Runs'] * 100, 2)

df['Boundary %'] = round(df['TotalBoundaries'] / df['BF'] * 100, 2)

df['NonBoundaryStrikeRate'] = round(df['NonBoundaryRuns'] / df['NonBoundaryBalls'] * 100, 2)

df['Boundary6 %'] = round(df['6s'] / (df['6s'] + df['4s']) * 100, 2)
df.head() # Inspect new format 
fig = px.scatter(df, 

                 x='Boundary %', 

                 y='NonBoundaryStrikeRate', 

                 color='Country', 

                 hover_name='Player', 

                 size='Runs')



fig.update_layout(

    height=500,

    title_text='ICC Cricket World Cup 2019 - Boundary Impact'

)

fig.show()
fig = px.scatter(df, 

                 x='Boundary %', 

                 y='Boundary6 %', 

                 color='Country', 

                 hover_name='Player', 

                 size='Runs')



fig.update_layout(

    height=500,

    title_text='ICC Cricket World Cup 2019 - 6 Hitting Impact'

)

fig.show()
grouping_columns = ['SR', 'RunsFromBoundary %', 'Boundary %', 'NonBoundaryStrikeRate', 'Boundary6 %']

df_chosen = df.loc[:,grouping_columns]
df_scaled = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df_chosen))

df_scaled.columns = grouping_columns

df_scaled.head()
# Instantiate a model with 3 centers

kmeans = KMeans(3)



# Then fit the model to your data using the fit method

model = kmeans.fit(df_scaled)



# Finally predict the labels on the same data to show the category that point belongs to

labels = model.predict(df_scaled)
model
labels_group = pd.Series(labels, dtype="category").map({0:'A', 1:'B',2:'C'})

df['Batting Classification'] = labels_group
fig = px.scatter(df, 

                 x='Boundary %', 

                 y='NonBoundaryStrikeRate', 

                 color='Batting Classification', 

                 hover_name='Player', 

                 size='Runs')



fig.update_layout(

    height=500,

    title_text='ICC Cricket World Cup 2019 - Batting Classifications'

)

fig.show()
df_pair = df_chosen.copy()

df_pair['Batting Classification'] = labels_group

sns.pairplot(df_pair, hue='Batting Classification')

plt.show()
scores = []

clusters = [x for x in range(2,10)]

for i in clusters:

    kmeans = KMeans(i)

    model = kmeans.fit(df_scaled)

    scores.append(np.abs(model.score(df_scaled)))

    

plt.plot(clusters, scores)  

plt.show()
from sklearn.metrics import silhouette_score



sil = []



# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2

for k in clusters:

  kmeans = KMeans(n_clusters = k).fit(df_scaled)

  labels = kmeans.labels_

  sil.append(silhouette_score(df_scaled, labels, metric = 'euclidean'))



plt.plot(clusters, sil)

plt.show()
kmeans = KMeans(8)

model = kmeans.fit(df_scaled)

labels = model.predict(df_scaled)

df_8_clusters = df.copy()

labels_group = pd.Series(labels, dtype="category").map({0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H'})

df_8_clusters['Batting Classification'] = labels_group

fig = px.scatter(df_8_clusters, 

                 x='Boundary %', 

                 y='NonBoundaryStrikeRate', 

                 color='Batting Classification', 

                 hover_name='Player', 

                 size='Runs')



fig.update_layout(

    height=500,

    title_text='ICC Cricket World Cup 2019 - Batting Classifications'

)

fig.show()
df_pair = df_chosen.copy()

df_pair['Batting Classification'] = labels_group

sns.pairplot(df_pair, hue='Batting Classification')

plt.show()