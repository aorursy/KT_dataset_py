# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# df_rank = pd.read_csv("/kaggle/input/billboard-ranking/billboardranking.csv")

# df_rank
df = pd.read_csv("/kaggle/input/billboard-100/billboard100 (1).csv", 

                 usecols=[0, 1, 2,15, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

df.rename(columns={"billboard": "year"}, inplace=True)

df.head()
# df = df.merge(df_rank, how="left", left_on=["title", "year"], right_on=["Song Title", "Year"])
df.shape
df.isna().sum().plot.bar()
df[df['top genre'].isna()]
df.loc[791, 'top genre'] = 'hip hop'

df.loc[548, 'top genre'] = 'modern alternative rock'

df.loc[658, 'top genre'] = 'modern alternative rock'

df.loc[595, 'top genre'] = 'hip hop'

df[df['top genre'].isna()]
features = ['bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur', 'acous', 'spch', 'pop']

df_norm = df.copy()

df_norm[features] = (df[features] - df[features].mean()) / df[features].std()

df_norm.head()
df['year'].unique()
# Compute the correlation matrix

corr = df_norm[features].corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5});
fig, axes = plt.subplots(5, 2, figsize=(15, 20), constrained_layout=True)

axes = axes.flatten()

for feat, ax in zip(features, axes):

    sns.lineplot(x="year", 

                 y=feat, 

                 data=df, ax=ax)

    ax.set_title(feat)

fig.show()
top_5_artists = df_norm['artist'].value_counts().head(5).index



for artist in top_5_artists:

    fig, axes = plt.subplots(5, 2, figsize=(15, 20), constrained_layout=True)

    axes = axes.flatten()

    df[artist] = df['artist'] == artist

    for feat, ax in zip(features, axes):

        sns.lineplot(x="year", 

                     y=feat, 

                     hue=artist,

                     data=df, ax=ax, ci='sd')

        ax.set_title(feat)

    fig.suptitle(artist, fontsize=20)

    fig.show()
for artist in top_5_artists:

    fig, axes = plt.subplots(5, 2, figsize=(15, 20), constrained_layout=True, sharey=True)

    axes = axes.flatten()

    df_norm[artist] = df_norm['artist'] == artist

    for feat, ax in zip(features, axes):

        sns.lineplot(x="year", 

                     y=feat, 

                     hue=artist,

                     data=df_norm, ax=ax, ci='sd')

        ax.set_title(feat)

    fig.suptitle(artist, fontsize=20)

    fig.show()
hm = df_norm[features].mean().to_frame(name="2010 - 2019").T

grid_kws = {"height_ratios": (.05, .9), "hspace": .3}

f, (cbar_ax, ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize = (10, 3))

ax = sns.heatmap(hm, ax=ax,

                 cbar_ax=cbar_ax,

                 cbar_kws={"orientation": "horizontal"})

ax.set_ylabel("", rotation=0)
hm = df_norm.groupby("year")[features].mean()

grid_kws = {"height_ratios": (.05, .9), "hspace": .3}

f, (cbar_ax, ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize = (10, 10))

ax = sns.heatmap(hm, ax=ax,

                 cbar_ax=cbar_ax,

                 cbar_kws={"orientation": "horizontal"},

                 linewidths=1)

ax.set_ylabel("", rotation=0)
df_norm.groupby("artist").count()['title']
hm = df_norm.groupby("artist")[features].median()

hm['count'] = df_norm.groupby("artist").count()['title']

hm['count'] = (hm['count']-hm['count'].mean())/hm['count'].std()

hm = hm.sort_values("count", ascending=False)

hm
grid_kws = {"height_ratios": (.05, .9), "hspace": .3}

f, (cbar_ax, ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize = (10, 100), constrained_layout=True)

ax = sns.heatmap(hm, ax=ax,

                 cbar_ax=cbar_ax,

                 cbar_kws={"orientation": "horizontal"}, cmap="coolwarm", linewidth=0.5, center=0)

ax.set_ylabel("", rotation=0)
hm = df_norm.groupby("top genre")[features].median()

hm['count'] = df_norm.groupby("top genre").count()['title']

hm['count'] = (hm['count']-hm['count'].mean())/hm['count'].std()

hm = hm.sort_values("count", ascending=False)





grid_kws = {"height_ratios": (.05, .9), "hspace": .3}

f, (cbar_ax, ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize = (10, 100), constrained_layout=True)

ax = sns.heatmap(hm, ax=ax,

                 cbar_ax=cbar_ax,

                 cbar_kws={"orientation": "horizontal"}, cmap="coolwarm", linewidth=0.5, center=0)

ax.set_ylabel("", rotation=0)