import pandas as pd

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

import seaborn as sns

import matplotlib.pyplot as plt

from math import sqrt

%matplotlib inline
# Loading data

df = pd.read_csv('../input/Pokemon.csv')



# Renaming one column for clarity

columns = df.columns.tolist()

columns[0] = 'id'

df.columns = columns



# Selecting columns to consider

cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']



df.head()
scaler = StandardScaler().fit(df[cols])

df_scaled = scaler.transform(df[cols])



print(df_scaled[:,0].mean())  # zero (or very close)

print(df_scaled[:,0].std())  # 1 (or very close)
pca = PCA(n_components=0.8)  # consider enough components to explain 80% of the variance

pca.fit(df_scaled)

pcscores = pd.DataFrame(pca.transform(df_scaled))

pcscores.columns = ['PC'+str(i+1) for i in range(len(pcscores.columns))]

loadings = pd.DataFrame(pca.components_, columns=cols)

loadings.index = ['PC'+str(i+1) for i in range(len(pcscores.columns))]
load_sqr = loadings**2

ax = sns.heatmap(load_sqr.transpose(), linewidths=0.5, cmap="BuGn", annot=True)

ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=8)

ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=8)
ax = sns.heatmap(loadings.transpose(), center=0, linewidths=0.5, 

                 cmap="RdBu", vmin=-1, vmax=1, annot=True)

ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=8)

ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=8)
# Create labels based on Type 1

labels = set(df['Type 1'])

df['type'] = df['Type 1']

lab_dict = dict()

for i, elem in enumerate(labels):

    lab_dict[elem] = i

df = df.replace({'type' : lab_dict})



pc_types = pcscores.copy()

pc_types['Type'] = df['Type 1']



# Biplots

def make_plot(pcscores, loadings, xval=0, yval=1, max_arrow=0.2, alpha=0.4):

    n = loadings.shape[1]

    scalex = 1.0 / (pcscores.iloc[:, xval].max() - pcscores.iloc[:, xval].min())  # Rescaling to be from -1 to +1

    scaley = 1.0 / (pcscores.iloc[:, yval].max() - pcscores.iloc[:, yval].min())



    pcscores.iloc[:, xval] = pcscores.iloc[:, xval] * scalex

    pcscores.iloc[:, yval] = pcscores.iloc[:, yval] * scaley



    g = sns.lmplot(x='PC{}'.format(xval + 1), y='PC{}'.format(yval + 1), hue='Type', data=pcscores,

                   fit_reg=False, size=6, palette='muted')



    for i in range(n):

        # Only plot the longer ones

        length = sqrt(loadings.iloc[xval, i] ** 2 + loadings.iloc[yval, i] ** 2)

        if length < max_arrow:

            continue



        plt.arrow(0, 0, loadings.iloc[xval, i], loadings.iloc[yval, i], color='k', alpha=0.9)

        plt.text(loadings.iloc[xval, i] * 1.15, loadings.iloc[yval, i] * 1.15,

                 loadings.columns.tolist()[i], color='k', ha='center', va='center')



    g.set(ylim=(-1, 1))

    g.set(xlim=(-1, 1))
# Actually make a biplot (PC3 vs PC4)

make_plot(pc_types, loadings, 2, 3, max_arrow=0.3)
best = pc_types.sort_values(by='PC4', ascending=False)[:2]

df.loc[best.index]
# Top HP Pokemon:

df.sort_values(by='HP', ascending=False)[:2]
g = sns.pairplot(pc_types, hue='Type', palette='muted')
print(pc_types.sort_values(by='PC2', ascending=False)[:1])

print(pc_types.sort_values(by='PC3', ascending=False)[:1])
df.loc[230]
make_plot(pc_types, loadings, 1, 2, max_arrow=0.3)