# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from glob import glob

import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

%matplotlib inline

from IPython.display import Audio, display, HTML

import itertools

from tqdm.notebook import tqdm



from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import umap.umap_ as umap

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from librosa import load



def standardize(f):

    return StandardScaler( # z = (x - mean) / stddev

        #with_std=False, #z = (x - mean) if False

        #with_mean=False,#z = x/stddev if False

    ).fit_transform(f)



def get_pca(features, n_components=2):

    pca = PCA(n_components=n_components)

    comp = pca.fit(features).transform(standardize(features))

    return MinMaxScaler().fit_transform(comp)







def get_tsne_embeddings(features, n_components=2, perplexity=15, iteration=20000):

    tsne_embedding = TSNE(n_components=n_components,

                     perplexity=perplexity,

                     verbose=0,

                     n_iter=iteration).fit_transform(standardize(features))

    return MinMaxScaler().fit_transform(tsne_embedding)





def get_umap_embeddings(features, n_components=2, neighbor=15, distance=0.1):

    umap_embedding = umap.UMAP(n_neighbors=neighbor,

                               min_dist=distance,

                               n_epochs=5000,

                               metric='correlation', 

                               n_components=n_components, 

                               verbose=False).fit_transform(standardize(features))

    return MinMaxScaler().fit_transform(umap_embedding)
df = pd.read_csv("../input/gtzan-dataset-music-genre-classification/Data/features_3_sec.csv")

df = df[df.filename.apply(lambda x: x.split(".")[-2]=='0')].copy().reset_index(drop=True)

features = df.iloc[:,2:-1]

le = LabelEncoder()

target = le.fit_transform(df.iloc[:,-1])

df.sample(5)
tsne_embeddings = dict()

perplexities = [10,15,20,25,30]

iterations = [5000,10000,15000,20000,25000]

for perplexity, iteration in tqdm(list(itertools.product(perplexities,iterations))): #for a prettier progress bar

    tsne_embedding = get_tsne_embeddings(features,

                                         perplexity=perplexity,

                                         iteration=iteration)

    tsne_embeddings[f"{perplexity}_{iteration}"]= tsne_embedding
color_scheme = {

 'classical': '#FE88FC',

 'jazz': '#F246FE',

 'blues': '#BF1CFD', 

 'metal': '#6ECE58',

 'rock': '#35B779',

 'disco': '#1F9E89',

 'pop': '#Fb9B06',

 'reggae': '#ED6925',

 'hiphop': '#CF4446',   

 'country': '#000004',   

 }



def plot_components(embeddings, n_rows, n_cols, title, suptitle):

    """helper function to plot embeddings"""

    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(20,20))

    r=0

    c=0

    for i in embeddings:

        df_ = pd.DataFrame(embeddings[i])

        df_.columns = [f'pc{i}' for i in range(1, len(df_.columns)+1)]

        df_['genre'] = df.iloc[:,-1]

        genres = df_.genre.unique()

        if c>n_cols-1: c=0; r+=1

        if r> n_rows-1: break

        for genre in genres:

            ax[r,c].scatter(df_[df_.genre==genre]['pc1'],

                            df_[df_.genre==genre]['pc2'], 

                            color=color_scheme[genre],

                            s=1

                           )

        ax[r,c].set_title(title.format(i.split('_')[0],i.split('_')[1]))

        ax[r,c].axis('off')

        c+=1



    plt.figlegend(df_.genre.unique(), loc='center right', prop={'size': 18}, markerscale=10);

    plt.suptitle(suptitle,fontsize=24);
plot_components(tsne_embeddings, len(perplexities), len(iterations), "perplexity={}\nepochs={}","t-SNE")
umap_embeddings = dict()

neighbors = list(range(10,31,5))

distances = [0.01, 0.02, 0.03, 0.04, 0.05]

for neighbor, distance in tqdm(list(itertools.product(neighbors,distances))):#for a prettier progress bar

    umap_embedding = get_umap_embeddings(features,

                                         neighbor=neighbor,

                                         distance=distance)

    umap_embeddings[f"{neighbor}_{distance}"]= umap_embedding
plot_components(umap_embeddings, len(neighbors), len(distances), "neighbors={}\nmin distance={}","UMAP")
tsne_embedding = get_tsne_embeddings(features,

                                     n_components=3,

                                     perplexity=30,

                                     iteration=20000)

df_tsne = pd.DataFrame(tsne_embedding)

df_tsne.columns = ['pc1','pc2','pc3']

df_tsne['genre'] = df.iloc[:,-1]

df_tsne['text'] = df[['filename','label']].apply(lambda x: f'{x[0]}<br>{x[1]}', axis=1)
import plotly.graph_objects as go



data = []

for g in df_tsne.genre.unique():

    trace = go.Scatter3d(

    x=df_tsne[df_tsne.genre==g].pc1.values,

    y=df_tsne[df_tsne.genre==g].pc2.values,

    z=df_tsne[df_tsne.genre==g].pc3.values,

    mode='markers',

    text=df_tsne[df_tsne.genre==g].text.values,

    hoverinfo = 'text',

    name=g,

    marker=dict(

            size=3,

            color=color_scheme[g],                

            opacity=0.9,

        )

    )

    data.append(trace)

fig = go.Figure(data=data)



fig.update_layout(title=f'TSNE', autosize=False,

                      width=600, height=600,

                      margin=dict(l=50, r=50, b=50, t=50),

                      scene=dict(xaxis=dict(title='pc1'), yaxis=dict(title='pc2'), zaxis=dict(title='pc3'))

                     )

fig.show()
umap_embedding = get_umap_embeddings(features,

                                     n_components=3,

                                     neighbor=20,

                                     distance=0.05)

df_umap = pd.DataFrame(umap_embedding)

df_umap.columns = ['pc1','pc2','pc3']

df_umap['genre'] = df.iloc[:,-1]

df_umap['text'] = df[['filename','label']].apply(lambda x: f'{x[0]}<br>{x[1]}', axis=1)

df_umap['color'] = df_umap.genre.apply(lambda x: color_scheme[x])
data = []

for g in df_umap.genre.unique():

    trace = go.Scatter3d(

    x=df_umap[df_umap.genre==g].pc1.values,

    y=df_umap[df_umap.genre==g].pc2.values,

    z=df_umap[df_umap.genre==g].pc3.values,

    mode='markers',

    text=df_umap[df_umap.genre==g].text.values,

    hoverinfo = 'text',

    name=g,

    marker=dict(

            size=3,

            color=color_scheme[g],                

            opacity=0.9,

        )

    )

    data.append(trace)

fig = go.Figure(data=data)



# tight layout

fig.update_layout(title=f'UMAP (Unsupervised)', autosize=False,

                      width=600, height=600,

                      margin=dict(l=50, r=50, b=50, t=50),

                      scene=dict(xaxis=dict(title='pc1'), yaxis=dict(title='pc2'), zaxis=dict(title='pc3'))

                     )

fig.show()
# pick a random point

a_pt = umap_embedding[np.random.randint(0,len(umap_embedding))]

# compute euclidean distances from all points to this point. Get sorted indices of top 6. The first one being the original point.

idx = np.argsort(np.linalg.norm(umap_embedding-a_pt, axis=1))[:6]



# display and play

path = "../input/gtzan-dataset-music-genre-classification/Data/genres_original/{}/{}.wav"

for i, k in enumerate(idx):

    # "filename" in the dataset is not a real file. Following lines involve some skulduggery to get correct audio segment from file. 

    fname = path.format(df.iloc[k,:]['label'],".".join(df.iloc[k,:]['filename'].split(".")[:-2]))

    segment_index = int(df.iloc[k,:]['filename'].split(".")[-2])

    y, sr = load(fname, mono=True)

    start = segment_index*sr

    end = start+(sr*3)

    y = y[start:end]

    

    if i==0: display(HTML(f"<p>Original:{df.iloc[k,:]['label']}</p><p>{fname}</p>"), Audio(y, rate=sr))

    else: display(HTML(f"<p>Neighbor {i}:{df.iloc[k,:]['label']}</p><p>{fname}</p>"), Audio(y, rate=sr))
umap_embedding = umap.UMAP(n_neighbors=30,

                               min_dist=0.4,

                               metric='correlation', 

                               n_components=3,

                               set_op_mix_ratio=0.25,

                               verbose=False).fit_transform(standardize(features), y=target)

umap_embedding = MinMaxScaler().fit_transform(umap_embedding)

df_umap = pd.DataFrame(umap_embedding)

df_umap.columns = ['pc1','pc2','pc3']

df_umap['genre'] = df.iloc[:,-1]

df_umap['text'] = df[['filename','label']].apply(lambda x: f'{x[0]}<br>{x[1]}', axis=1)
data = []

for g in df_umap.genre.unique():

    trace = go.Scatter3d(

    x=df_umap[df_umap.genre==g].pc1.values,

    y=df_umap[df_umap.genre==g].pc2.values,

    z=df_umap[df_umap.genre==g].pc3.values,

    mode='markers',

    text=df_umap[df_umap.genre==g].text.values,

    hoverinfo = 'text',

    name=g,

    marker=dict(

            size=3,

            color=color_scheme[g],                

            opacity=0.9,

        )

    )

    data.append(trace)

fig = go.Figure(data=data)



# tight layout

fig.update_layout(title=f'UMAP (Supervised)', autosize=False,

                      width=600, height=600,

                      margin=dict(l=50, r=50, b=50, t=50),

                      scene=dict(xaxis=dict(title='pc1'), yaxis=dict(title='pc2'), zaxis=dict(title='pc3'))

                     )

fig.show()