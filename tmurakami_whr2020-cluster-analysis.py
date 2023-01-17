!pip install pyclustering
!pip install folium
!pip install country_converter
import json
import folium
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import country_converter as coco
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.xmeans import (
    xmeans, 
    kmeans_plusplus_initializer,
)
from pathlib import Path
df = pd.read_csv('/kaggle/input/world-happiness-report/2020.csv')
print(df.isnull().sum(axis=0))
print(f'shape: {df.shape}')
columns = list(df.columns.values)
for i in range(6,12):
    print(columns[i])
    x = 1*df[columns[i]].values if i != 11 else -df[columns[i]].values
    dystopia = min(x)
    beta = max(df[columns[i+7]].values)/max(x-dystopia)
    print(f"  dystopia: {dystopia}")
    print(f"  beta: {beta}")
    plt.plot(beta*(x-dystopia), df[columns[i+7]].values, label=f"{columns[i]} vs. {columns[i+7]}")
    plt.xlabel("value")
    plt.ylabel("Explained by value")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.show
    # beta => Table 11 and dystopia => Table 20
dystopia = np.average(df[columns[-1]].values)
residual = df[columns[-1]].values - dystopia
plt.plot(residual)

dic = {-i:np.array([0.,0.]) for i in range(2,8)}
for year in [2017, 2018, 2019]:
    _df = pd.read_csv(f'/kaggle/input/world-happiness-report/{year}.csv')
    _columns = _df.columns.values
    for i in range(6):
        dic[-(i+2)] += np.array([np.sum(_df[_columns[-(i+2)]].values), len(_df[_columns[-(i+2)]].values)])
    
_residual = np.zeros(len(residual))
for i in range(2,8):
    _residual +=  dic[-i][0]/dic[-i][1] - df[columns[-i]].values
plt.plot(_residual)

_residual = np.zeros(len(residual))
for i in range(2,8):
    _residual +=  np.average(df[columns[-i]].values) - df[columns[-i]].values
plt.plot(_residual)

plt.show()
print(np.average(residual))
plt.plot(np.sum(df[columns[-7:]].values, axis=1), df[columns[2]].values)
plt.ylabel(columns[2])
plt.xlabel("\n + ".join(columns[-7:]))
plt.show()
# Country name
names = df[columns[0]].values
# Variables from "Explained by: Log GDP per capita" to "Dystopia + residual" 
variables = df[columns[-7:]].values
# Ladder score
ladder = df[columns[2]].values
# Standarization
scaler = StandardScaler()
scaled_variables = scaler.fit_transform(variables)
def annotate_coef(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(
        f'r = {r:.2f}',
        xy=(.1, .9),
        xycoords=ax.transAxes
    )

g = sns.PairGrid(df[[columns[2]]+columns[-7:]])
g.map_upper(plt.scatter, marker='.')
g.map_diag(sns.distplot, kde=False)
g.map_lower(sns.kdeplot)
g.map_lower(annotate_coef)

plt.show()
# X-means
model = xmeans(
    scaled_variables,
    kmeans_plusplus_initializer(scaled_variables, 2).initialize()
)
model.process()

# X-means clustering
colors = [0]*len(names)
clusters = model.get_clusters()
for cluster_id,cluster in enumerate(clusters):
    for i in cluster:
        colors[i] = cluster_id

# Distance from the center of the cluster
distances = [0]*len(names)
centers = model.get_centers()
for cluster_id,cluster in enumerate(clusters):
    for i in cluster:
        distances[i] = np.linalg.norm(scaled_variables[i]-centers[cluster_id])
# PCA
model = PCA()
pca_variables = model.fit_transform(scaled_variables)
pca_centers = model.transform(centers)

# Principal components
print(
    pd.DataFrame(
        model.components_,
        index=columns[-7:],
        columns=[f'PC{i+1}' for i in range(model.n_components_)]
    )
)

# Proportion of variance and Cumulative proportion
print(
    pd.DataFrame(
        np.matrix([
            model.explained_variance_ratio_,
            np.cumsum(model.explained_variance_ratio_)
        ]).T,
        index=[f'PC{i+1}' for i in range(model.n_components_)],
        columns=['Proportion of variance', 'Cumulative proportion']
    )
)
cmap = plt.cm.Set1

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_title(f'{len(set(colors))} clusters')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.scatter(
    pca_variables[:,0],
    pca_variables[:,1],
    pca_variables[:,2],
    marker=".",
    c=[cmap(cluster_id) for cluster_id in colors],
)
for cluster_id in range(len(pca_centers)):
    ax.scatter(
        pca_centers[cluster_id,0],
        pca_centers[cluster_id,1],
        pca_centers[cluster_id,2],
        marker="o",
        c=cmap(cluster_id),
        alpha=0.5,
        linewidth=0,
        s=200,
        label=f'Center of cluster {cluster_id+1}',
)
ax.view_init(90, 0)
ax.legend(bbox_to_anchor=(1,1), loc='upper left')

plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.spines['top'].set_alpha(0)
ax.spines['bottom'].set_alpha(0)
ax.spines['right'].set_alpha(0)
ax.spines['left'].set_alpha(0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Distance from the center', labelpad=20)
ax.set_ylabel('Frequency', labelpad=20)
for cluster_id,cluster in enumerate(clusters):
    ax = fig.add_subplot(1,3,cluster_id+1)
    ax.set_title(f'Cluster {cluster_id+1}')
    ax.hist([distances[i] for i in cluster], bins=50, color=cmap(cluster_id))
fig.tight_layout()

plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
for cluster_id, cluster in enumerate(clusters):
    ax.plot(
        centers[cluster_id],
        color=cmap(cluster_id),
        linewidth=0.5,
        markersize=2,
        marker='o',
        label=f'Center of cluster {cluster_id+1}',
    )
    """
    for i in cluster:
        ax.plot(
            scaled_variables[i],
            color=cmap(cluster_id),
            linewidth=0.5,
            markersize=0,
            alpha=0.5,
        )
    """
    std = np.std([scaled_variables[i] for i in cluster], axis=0)
    ax.fill_between(
        range(len(centers[cluster_id])),
        centers[cluster_id]+std, 
        centers[cluster_id]-std, 
        color=cmap(cluster_id), 
        alpha=0.2,
    )
ax.set_xticks(range(len(columns[-7:])))
ax.set_xticklabels(
    [x.replace(':', ':\n') for x in columns[-7:]],
    rotation=90
)
ax.set_ylabel('Value')
ax.legend(bbox_to_anchor=(1,1), loc='upper left')

plt.show()
fig = plt.figure(figsize=(5, 30))
ax = fig.add_subplot(111)
bar_size = 0.8
N = len(names)
ax.barh(
    range(N),
    ladder,
    color=[cmap(cluster_id) for cluster_id in colors],
    height = bar_size, 
    align='center',
)
ax.set_yticks(range(N))
ax.set_yticklabels(names)
ax.set_ylim([-bar_size*0.5, N-1+bar_size*0.5])
ax.invert_yaxis()
ax.set_xlabel(columns[2])
ax.set_title('Ladder score ranking')

plt.show()
def annotate_coef(x, y, **kwargs):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(
        f'r = {r:.2f}',
        xy=(.1, .9),
        xycoords=ax.transAxes
    )
    
_df = df[[columns[2]]+columns[-7:]]
_df['cluster_id'] = [f"{cluster_id}" for cluster_id in colors]

sns.set_palette(sns.color_palette([cmap(i) for i in range(len(clusters))]))

g = sns.PairGrid(_df, vars=[columns[2]]+columns[-7:], hue="cluster_id")
g.map_upper(plt.scatter, marker='.')
g.map_diag(sns.distplot, kde=False)
g.map_lower(sns.kdeplot)
g.map_lower(annotate_coef)

plt.show()
# Load GeoJSON
with Path('/kaggle/input/worldgeojson/custom.geo.json').open('r') as f:
    geojson = json.load(f)
# Add "Cluster id" column to DataFrame
df['Cluster id'] = colors

# Convert country name to ISO Alpha-3 code
df['ISO3'] = coco.convert(names=list(df['Country name'].values), to='ISO3')
pd.set_option('display.max_rows', 200)

# Modify the code of North Cyprus
df.loc[df['Country name'] == 'North Cyprus', 'ISO3'] = 'CYN'
def get_choropleth(target_column):
    fmap = folium.Map(
        tiles='Mapbox Bright',
        location=[40, 0],
        zoom_start=2.25,
    )
    
    choropleth = folium.Choropleth(
        data=df,
        columns=['ISO3', target_column],
        legend_name=target_column,
        line_opacity=0.2,
        highlight=True,
        geo_data=geojson,
        key_on='feature.properties.gu_a3',
    ).add_to(fmap)

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['name'], labels=False)
    )
    
    return fmap
import branca.colormap as bcm
from collections import defaultdict

fmap = folium.Map(
    tiles='Mapbox Bright',
    location=[40, 0],
    zoom_start=2.25,
)

stepcolor = bcm.StepColormap(
    ['black']+[cmap(i) for i in range(len(clusters))],
    vmin=0, vmax=3,
    index=[-1, 0, 1, 2, 3],
    caption='Cluster id'
).add_to(fmap)

colordict = defaultdict(lambda: -1)
for (k,v) in df.set_index('ISO3')['Cluster id'].to_dict().items():
    colordict[k] = v

folium.GeoJson(
    geojson,
    style_function=lambda feature:{
        'fillColor':stepcolor(colordict[str(feature['properties']['gu_a3'])]+1),
        'color':'#000000',
        'fillOpacity': 0.5,
        'weight': 0.2
    },
    highlight_function=lambda feature:{
        'color':'#000000',
        'fillOpacity': 0.3,
        'weight': 0.2
    },
    tooltip=folium.features.GeoJsonTooltip(
        fields=['name'],
        aliases=['Country name: '],
        style=(" ".join([
            "background-color: white;",
            "color: #333333;",
            "font-family: arial;",
            "font-size: 12px;",
            "padding: 10px;"
        ]))
    )
).add_to(fmap)

fmap

def get_choropleth(target_column):
    fmap = folium.Map(
        tiles='Mapbox Bright',
        location=[40, 0],
        zoom_start=2.25,
    )
    
    choropleth = folium.Choropleth(
        data=df,
        columns=['ISO3', target_column],
        legend_name=target_column,
        fill_color='YlOrRd',
        line_opacity=0.2,
        highlight=True,
        geo_data=geojson,
        key_on='feature.properties.gu_a3',
    ).add_to(fmap)

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['name'], labels=False)
    )
    
    return fmap
get_choropleth('Ladder score')
get_choropleth(columns[-7])
get_choropleth(columns[-6])
get_choropleth(columns[-5])
get_choropleth(columns[-4])
get_choropleth(columns[-3])
get_choropleth(columns[-2])
get_choropleth(columns[-1])