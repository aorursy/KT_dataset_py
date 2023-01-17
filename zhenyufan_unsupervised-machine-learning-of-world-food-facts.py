import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import matplotlib.cbook

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected = True)

import plotly.graph_objs as go

%matplotlib inline



warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)



import nltk

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

import gensim

from gensim import corpora



from sklearn.cluster import KMeans

from sklearn.cluster import MiniBatchKMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from yellowbrick.cluster import KElbowVisualizer

from yellowbrick.cluster import SilhouetteVisualizer
df = pd.read_csv("../input/nutrition_table.csv")

df.drop(["Unnamed: 0", "exceeded", "g_sum", "energy_100g"], axis=1, inplace=True) #drop two rows we don't need

df = df.drop(df.index[[1,11877]]) #drop outlier

df.rename(index=str, columns={"reconstructed_energy": "energy_100g"}, inplace=True)

df.head()
df.columns
warnings.filterwarnings("ignore")

sns.set(style='white', palette='muted', color_codes=True)



f, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True)

sns.despine(left=True)



sns.distplot(df.fat_100g, color='b', ax=axes[0, 0])

sns.distplot(df.carbohydrates_100g, color='g', ax=axes[0, 1])

sns.distplot(df.sugars_100g, color='r', ax=axes[1, 0])

sns.distplot(df.proteins_100g, color='m', ax=axes[1, 1])

plt.tight_layout()
high_fat_df = df[df.fat_100g > df.fat_100g.quantile(.98)]

high_fat_text = high_fat_df['product'].values



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(high_fat_text))



fig = plt.figure(

    figsize = (10, 7),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)
high_carbohydrate_df = df[df.carbohydrates_100g > df.carbohydrates_100g.quantile(.98)]

high_carbohydrate_text = high_carbohydrate_df['product'].values



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(high_carbohydrate_text))



fig = plt.figure(

    figsize = (10, 7),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)
high_sugar_df = df[df.sugars_100g > df.sugars_100g.quantile(.98)]

high_sugar_text = high_sugar_df['product'].values



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(high_sugar_text))



fig = plt.figure(

    figsize = (10, 7),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)
high_protein_df = df[df.proteins_100g > df.proteins_100g.quantile(.98)]

high_protein_text = high_protein_df['product'].values



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(high_protein_text))



fig = plt.figure(

    figsize = (10, 7),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)
high_salt_df = df[df.salt_100g > df.salt_100g.quantile(.98)]

high_salt_text = high_salt_df['product'].values



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(high_salt_text))



fig = plt.figure(

    figsize = (10, 7),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)
high_energy_df = df[df.energy_100g > df.energy_100g.quantile(.98)]

high_energy_text = high_energy_df['product'].values



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(high_energy_text))



fig = plt.figure(

    figsize = (10, 7),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)
dataset = df['product'].fillna("").values

raw_text_data = [d.split() for d in dataset]
stop = stopwords.words('english')
text_data = [item for item in raw_text_data if item not in stop]
from gensim import corpora

dictionary = corpora.Dictionary(text_data)

corpus = [dictionary.doc2bow(text) for text in text_data]
import gensim

NUM_TOPICS = 5

ldamodel = gensim.models.ldamodel.LdaModel(

    corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)



topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
df_cluster_features = df.drop("product", axis=1)

scaler = MinMaxScaler()

scaler.fit(df_cluster_features)

scaler.transform(df_cluster_features)
model = KMeans()

visualizer = KElbowVisualizer(model, k=(1,11))



visualizer.fit(df_cluster_features)

visualizer.poof() 
model = MiniBatchKMeans(3)

visualizer = SilhouetteVisualizer(model)



visualizer.fit(df_cluster_features) 

visualizer.poof()
k_means = KMeans(n_clusters=3)

kmeans = k_means.fit(scaler.transform(df_cluster_features))

df['cluster'] = kmeans.labels_

df.head()
trace = go.Scatter3d(

    x=df['fat_100g'],

    y=df['carbohydrates_100g'],

    z=df['sugars_100g'],

    mode='markers',

    text=df['product'],

    marker=dict(

        size=12,

        color=df['cluster'],                

        colorscale='Viridis',

        opacity=0.8

    )

)



data = [trace]

layout = go.Layout(

    showlegend=False,

    title='Fat-Carb-Sugar:  Food Energy Types',

    scene = dict(

        xaxis = dict(title='X: Fat Content-100g'),

        yaxis = dict(title="Y:  Carbohydrate Content-100g"),

        zaxis = dict(title="Z:  Sugar Content-100g"),

    ),

    width=900,

    height=900,

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace = go.Scatter3d(

    x=df['fat_100g'],

    y=df['carbohydrates_100g'],

    z=df['proteins_100g'],

    mode='markers',

    text=df['product'],

    marker=dict(

        size=12,

        color=df['cluster'],                

        colorscale='Viridis',

        opacity=0.8

    )

)



data = [trace]

layout = go.Layout(

    showlegend=False,

    title='Fat-Carb-Protein:  Food Energy Types',

    scene = dict(

        xaxis = dict(title='X: Fat Content-100g'),

        yaxis = dict(title="Y:  Carbohydrate Content-100g"),

        zaxis = dict(title="Z:  Protein Content-100g"),

    ),

    width=900,

    height=900,

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace = go.Scatter3d(

    x=df['fat_100g'],

    y=df['carbohydrates_100g'],

    z=df['salt_100g'],

    mode='markers',

    text=df['product'],

    marker=dict(

        size=12,

        color=df['cluster'],                

        colorscale='Viridis',

        opacity=0.8

    )

)



data = [trace]

layout = go.Layout(

    showlegend=False,

    title='Fat-Carb-Salt:  Food Energy Types',

    scene = dict(

        xaxis = dict(title='X: Fat Content-100g'),

        yaxis = dict(title="Y:  Carbohydrate Content-100g"),

        zaxis = dict(title="Z:  Salt Content-100g"),

    ),

    width=900,

    height=900,

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace = go.Scatter3d(

    x=df['fat_100g'],

    y=df['carbohydrates_100g'],

    z=df['energy_100g'],

    mode='markers',

    text=df['product'],

    marker=dict(

        size=12,

        color=df['cluster'],                

        colorscale='Viridis',

        opacity=0.8

    )

)



data = [trace]

layout = go.Layout(

    showlegend=False,

    title='Fat-Carb-Energy:  Food Energy Types',

    scene = dict(

        xaxis = dict(title='X: Fat Content-100g'),

        yaxis = dict(title="Y:  Carbohydrate Content-100g"),

        zaxis = dict(title="Z:  n=Energy Content-100g"),

    ),

    width=900,

    height=900,

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace = go.Scatter3d(

    x=df['proteins_100g'],

    y=df['carbohydrates_100g'],

    z=df['salt_100g'],

    mode='markers',

    text=df['product'],

    marker=dict(

        size=12,

        color=df['cluster'],                # set color to an array/list of desired values

        colorscale='Viridis',   # choose a colorscale

        opacity=0.8

    )

)



data = [trace]

layout = go.Layout(

    showlegend=False,

    title='Protein, Carb, Salt:  Food Energy Types',

    scene = dict(

        xaxis = dict(title='X: Protein Content-100g'),

        yaxis = dict(title='Y: Carbohydrate Content-100g'),

        zaxis = dict(title='Z: Salt Content-100g'),

    ),

    width=1000,

    height=900,

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df_new = df.copy()

df_new.columns
df_new['calorie_100g'] = df['fat_100g'] + df['carbohydrates_100g'] + df['sugars_100g']

df_new_features = df_new.drop(['fat_100g', 'carbohydrates_100g', 'sugars_100g', 'cluster', 'product'], axis=1)

df_new_features.head()
scaler.fit(df_new_features)

scaler.transform(df_new_features)



model = KMeans()

visualizer = KElbowVisualizer(model, k=(1,11))



visualizer.fit(df_new_features)

visualizer.poof() 
k_means = KMeans(n_clusters=3)

kmeans = k_means.fit(scaler.transform(df_new_features))

df_new_features['cluster'] = kmeans.labels_

df_new_features.head()
trace = go.Scatter3d(

    x=df_new_features['proteins_100g'],

    y=df_new_features['salt_100g'],

    z=df_new_features['energy_100g'],

    mode='markers',

    marker=dict(

        size=12,

        color=df_new_features['cluster'],                # set color to an array/list of desired values

        colorscale='Viridis',   # choose a colorscale

        opacity=0.8

    )

)



data = [trace]

layout = go.Layout(

    showlegend=False,

    title='Protein, Salt, Energy:  Food Types',

    scene = dict(

        xaxis = dict(title='X: Protein Content-100g'),

        yaxis = dict(title='Y: Salt Content-100g'),

        zaxis = dict(title='Z: Energy Content-100g'),

    ),

    width=1000,

    height=900,

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df_features_calorie = df_new_features.copy()

df_features_calorie = df_features_calorie.join(df['product'])

df_features_calorie['text'] = df_features_calorie['cluster'].astype(str) + ' ' + df_features_calorie['product']

df_features_calorie.head()
trace = go.Scatter3d(

    x=df_features_calorie['proteins_100g'],

    y=df_features_calorie['salt_100g'],

    z=df_features_calorie['calorie_100g'],

    text=df_features_calorie['text'],

    mode='markers',

    marker=dict(

        size=12,

        color=df_features_calorie['cluster'],               

        colorscale='Viridis',   

        opacity=0.8

    )

)



data = [trace]

layout = go.Layout(

    showlegend=False,

    title='Protein, Salt, Calorie:  Food Types',

    scene = dict(

        xaxis = dict(title='X: Protein Content-100g'),

        yaxis = dict(title='Y: Salt Content-100g'),

        zaxis = dict(title='Z: Calorie Content-100g'),

    ),

    width=1000,

    height=900,

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df_final = df_features_calorie.drop(['energy_100g'], axis=1)

df_final = df_final.replace({'cluster':{0:'high calorie', 1:'low calorie', 2:'average calorie'}})

df_final.head()
df_cluster_features.head()
cf_std = StandardScaler().fit_transform(df_cluster_features)
cov_matrix = np.cov(cf_std.T)

cov_matrix
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
total = sum(eig_vals)

var_exp = [(i / total)*100 for i in sorted(eig_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)



trace1 = dict(

    type='bar',

    x=['PC %s' %i for i in range(1,7)],

    y=var_exp,

    name='Individual'

)



trace2 = dict(

    type='scatter',

    x=['PC %s' %i for i in range(1,7)], 

    y=cum_var_exp,

    name='Cumulative'

)



data = [trace1, trace2]



layout=dict(

    title='Explained variance by different principal components',

    yaxis=dict(

        title='Explained variance in percent'

    ),

    annotations=list([

        dict(

            x=1.16,

            y=1.05,

            xref='paper',

            yref='paper',

            text='Explained Variance',

            showarrow=False,

        )

    ])

)



fig = dict(data=data, layout=layout)

iplot(fig)
pca = PCA(n_components=4)

principal_components = pca.fit_transform(df_cluster_features)

# Principles' names come from eigenvectors. Those eigenvectors stands for different weights of original variables

pc_df = pd.DataFrame(data = principal_components

             , columns = ['chemical element orinted', 'high energy', 'diabetes friendly', 'hypertension friendly'])

pc_df.head()
scaler.fit(pc_df)

scaler.transform(pc_df)



model = KMeans()

visualizer = KElbowVisualizer(model, k=(1,11))



visualizer.fit(pc_df)

visualizer.poof() 
k_means = KMeans(n_clusters=3)

kmeans = k_means.fit(scaler.transform(pc_df))

pc_df['cluster'] = kmeans.labels_

pc_df.head()
trace = go.Scatter3d(

    x=pc_df['chemical element orinted'],

    y=pc_df['high energy'],

    z=pc_df['diabetes friendly'],

    text=pc_df['cluster'],

    mode='markers',

    marker=dict(

        size=12,

        color=pc_df['cluster'],              

        colorscale='Viridis',   

        opacity=0.8

    )

)



data = [trace]

layout = go.Layout(

    showlegend=False,

    title='Different Kinds of Food',

    scene = dict(

        xaxis = dict(title='X: Chemical element orinted'),

        yaxis = dict(title='Y: High energy'),

        zaxis = dict(title='Z: Diabetes friendly'),

    ),

    width=1000,

    height=900,

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace = go.Scatter3d(

    x=pc_df['chemical element orinted'],

    y=pc_df['high energy'],

    z=pc_df['hypertension friendly'],

    text=pc_df['cluster'],

    mode='markers',

    marker=dict(

        size=12,

        color=pc_df['cluster'],                

        colorscale='Viridis', 

        opacity=0.8

    )

)



data = [trace]

layout = go.Layout(

    showlegend=False,

    title='Different Kinds of Food',

    scene = dict(

        xaxis = dict(title='X: Chemical element orinted'),

        yaxis = dict(title='Y: High energy'),

        zaxis = dict(title='Z: Hypertension friendly'),

    ),

    width=1000,

    height=900,

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)