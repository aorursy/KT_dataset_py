# Instead of sklearn TSNE implementation I'll use MulticoreTSNE 

# beacuse it much much faster

!pip install MulticoreTSNE
# Importing libraries

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split



from mpl_toolkits.mplot3d import Axes3D



from MulticoreTSNE import MulticoreTSNE as TSNE



warnings.filterwarnings('ignore')

pd.set_option('max_columns', None)

pd.set_option('max_rows', 100)
# I want ot define a couple of functions to make my code more DRY

def gen_dataset(samples):

    '''

    Generates dataset with all fraud transactions and "samples" part of non_fraud transactions

    '''

    # Sub-sampling dataset

    fraud = df[df['Class'] == 1]

    non_fraud = df[df['Class'] == 0].sample(samples, random_state = 1)

    df_scaled = pd.concat([fraud, non_fraud])

    df_scaled = shuffle(df_scaled, random_state = 1)



    Y = df_scaled['Class']



    df_scaled = df_scaled.drop('Class', axis = 1)



    # Standardize the data to a mean of zero and a SD of 1 to remove the scale effect of the variables

    scaler = StandardScaler()

    df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns = df_scaled.columns)

    return (df_scaled, Y)



def plots(df_2d, df3d, t = False):

    '''

    Creates 2d and 3d plots

    '''

    fig = plt.figure(figsize = (20, 8))

    ax1 = fig.add_subplot(121)

    if t:

        sns.scatterplot(x = 'PC1', y = 'PC2', data = df_2d, hue = 'Class').set_title(f'TSNE with 2 components')

    else:

        sns.scatterplot(x = 'PC1', y = 'PC2', data = df_2d, hue = 'Class').set_title(f'PCA with 2 components\nexpl. variance: {variance_2d}%')



    ax2 = fig.add_subplot(122, projection = '3d')

    f = df3d[df3d['Class'] == 1]

    nf = df3d[df3d['Class'] == 0]

    ax2.scatter(nf['PC1'], nf['PC2'], nf['PC3'], marker = 'o')

    ax2.scatter(f['PC1'], f['PC2'], f['PC3'], marker = 'x')

    ax2.view_init(30, 240)

    if t:

        plt.title(('TSNE with 3 components'))

    else:

        plt.title((f'PCA with 3 components\nexpl. variance: {variance_3d}%'))

    ax2.set_xlabel('PC1')

    ax2.set_ylabel('PC2')

    ax2.set_zlabel('PC3')

    plt.show()
# Loading dataset

df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

print(df.shape)

df.head()
# Creating dataset for PCA

df_pca, Y = gen_dataset(50000)
# PCA with 2 components

pca_2d = PCA(n_components = 2, random_state = 1)

pca_df_2d = pca_2d.fit_transform(df_pca)

variance_2d = round(sum(pca_2d.explained_variance_ratio_) * 100, 2)

pca_df_2d = pd.DataFrame(pca_df_2d, columns = ['PC1', 'PC2'])

pca_df_2d['Class'] = Y.values



# PCA with 3 components

pca_3d = PCA(n_components = 3, random_state = 1)

pca_df_3d = pca_3d.fit_transform(df_pca)

variance_3d = round(sum(pca_3d.explained_variance_ratio_) * 100, 2)

pca_df_3d = pd.DataFrame(pca_df_3d, columns = ['PC1', 'PC2', 'PC3'])

pca_df_3d['Class'] = Y.values



plots(pca_df_2d, pca_df_3d)
# Creating dataset for TSNE

df_tsne, Y = gen_dataset(15000)
tsne_2d = TSNE(n_components = 2, 

            perplexity = 30, 

            early_exaggeration=12.0,

            learning_rate=200.0,

            n_iter=1000,

            random_state = 1, 

            n_jobs = -1)

tsne_df_2d = tsne_2d.fit_transform(df_tsne)



tsne_df_2d = pd.DataFrame(tsne_df_2d, columns = ['PC1', 'PC2'])

tsne_df_2d['Class'] = Y.values



tsne_3d = TSNE(n_components = 3, 

            perplexity = 30, 

            early_exaggeration=12.0,

            learning_rate=200.0,

            n_iter=1000,

            random_state = 1, 

            n_jobs = -1)

tsne_df_3d = tsne_3d.fit_transform(df_tsne)



tsne_df_3d = pd.DataFrame(tsne_df_3d, columns = ['PC1', 'PC2', 'PC3'])

tsne_df_3d['Class'] = Y.values



plots(tsne_df_2d, tsne_df_3d, t = True)