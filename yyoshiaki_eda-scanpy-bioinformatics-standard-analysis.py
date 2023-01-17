!pip install scanpy
!conda install -y -c conda-forge python-igraph leidenalg
import numpy as np

import pandas as pd

import scanpy as sc

from anndata import AnnData

import matplotlib.pyplot as plt

from tqdm import tqdm



from sklearn import preprocessing



sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)

sc.logging.print_header()

sc.settings.set_figure_params(dpi=80, facecolor='white')
def label_encoding(train: pd.DataFrame, test: pd.DataFrame, encode_cols):

    n_train = len(train)

    idx_train = train.index

    idx_test = test.index

    train = pd.concat([train, test], sort=False).reset_index(drop=True)

    for f in encode_cols:

        try:

            lbl = preprocessing.LabelEncoder()

            train[f] = lbl.fit_transform(list(train[f].values))

        except:

            print(f)

    test = train[n_train:].reset_index(drop=True)

    train = train[:n_train]

    train.index = idx_train

    test.index = idx_test

    return train, test
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv', 

                    index_col=0)

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv',

                  index_col=0)



train_target = pd.read_csv("../input/lish-moa/train_targets_scored.csv", 

                           index_col=0)



train, test = label_encoding(train, test, ['cp_type', 'cp_dose'])

train['dataset'] = 'train'

test['dataset'] = 'test'



df = pd.concat([train, test])



list_obs_col = ['cp_type', 'cp_time', 'cp_dose', 'dataset']

list_genes = [x for x in df.columns if x not in list_obs_col]



adata = AnnData(df[list_genes], obs=df[list_obs_col])

adata.obs = pd.concat([adata.obs, train_target.reindex(df.index)], 

                      axis=1)

adata
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

sc.pl.highly_variable_genes(adata)
adata.raw = adata

adata = adata[:, adata.var.highly_variable]
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color='g-0', size=10)
sc.pl.pca_variance_ratio(adata, log=True)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

sc.tl.umap(adata)
adata.obs
sc.pl.umap(adata, color=['g-0', 'dataset', '5-alpha_reductase_inhibitor',

                        '11-beta-hsd1_inhibitor', '11-beta-hsd1_inhibitor', 'acat_inhibitor',

                        'acetylcholine_receptor_agonist', 'acetylcholine_receptor_antagonist', 

                         'acetylcholinesterase_inhibitor'],

          size=10)
sc.tl.leiden(adata)

sc.pl.umap(adata, color='leiden')
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')

sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
sc.pl.umap(adata, color=['g-37', 'c-26', 'g-369', 'leiden'],

          size=10)
sc.pl.rank_genes_groups_matrixplot(adata, groupby='leiden', n_genes=3)
X_train = pd.concat([adata.obs.loc[train.index, ['cp_type', 'cp_time', 'cp_dose']].reset_index(drop=True), 

              pd.DataFrame(adata.obsm['X_pca'][adata.obs.loc[train.index].reset_index().index]),

              pd.DataFrame(adata.obsm['X_umap'][adata.obs.loc[train.index].reset_index().index]),

              pd.get_dummies(pd.DataFrame(adata.obs['leiden'][adata.obs.loc[train.index].reset_index().index])

                            ).drop('leiden_0', axis=1).reset_index(drop=True)],

         axis=1)



X_test = pd.concat([adata.obs.loc[test.index, ['cp_type', 'cp_time', 'cp_dose']].reset_index(drop=True), 

              pd.DataFrame(adata.obsm['X_pca'][adata.obs.loc[test.index].reset_index().index]),

              pd.DataFrame(adata.obsm['X_umap'][adata.obs.loc[test.index].reset_index().index]),

              pd.get_dummies(pd.DataFrame(adata.obs['leiden'][adata.obs.loc[test.index].reset_index().index])

                            ).drop('leiden_0', axis=1).reset_index(drop=True)],

             axis=1)
from sklearn.linear_model import LogisticRegression





def lr(X_train, y_train, X_test):

    reg = LogisticRegression().fit(X_train, y_train)

    return reg.predict_proba(X_test)[:,1]



list_pred = []

for c in train_target.columns:

    list_pred.append(lr(X_train, train_target[c], X_test))

df_pred = pd.DataFrame(list_pred).T

df_pred.index = test.index

df_pred.columns = train_target.columns

df_pred.to_csv('submission.csv')
adata.obs.loc[df_pred.index, df_pred.columns] = df_pred
adata[adata.obs['dataset'] == 'train']
sc.pl.umap(adata[adata.obs['dataset'] == 'train'], color=['5-alpha_reductase_inhibitor',

                        '11-beta-hsd1_inhibitor', '11-beta-hsd1_inhibitor', 'acat_inhibitor',

                        'acetylcholine_receptor_agonist', 'acetylcholine_receptor_antagonist', 

                         'acetylcholinesterase_inhibitor'],

          size=10)
sc.pl.umap(adata[adata.obs['dataset'] == 'test'], color=['5-alpha_reductase_inhibitor',

                        '11-beta-hsd1_inhibitor', '11-beta-hsd1_inhibitor', 'acat_inhibitor',

                        'acetylcholine_receptor_agonist', 'acetylcholine_receptor_antagonist', 

                         'acetylcholinesterase_inhibitor'],

          size=40)