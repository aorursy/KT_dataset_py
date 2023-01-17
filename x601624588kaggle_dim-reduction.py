# ! pip install scanpy
# ! pip install scvi
import scanpy as sc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import os
import matplotlib.pyplot as plt
dpath = r'../input/pbmc3k-filtered-gene-bc-matrices/pbmc3k_filtered_gene_bc_matrices/hg19'
adata = sc.read_10x_mtx(
    dpath,                     # the directory with the `.mtx` file
    var_names='gene_symbols',  # use gene symbols for the variable names (variables-axis index)
    cache=True)                # write a cache file for faster subsequent reading
adata
sc.pl.highest_expr_genes(adata, n_top=20)
#filter low-quality cells and empty droplets
sc.pp.filter_cells(adata, min_genes=200)
#filter low-level expressed genes
sc.pp.filter_genes(adata, min_cells=3)
# annotate the group of mitochondrial genes as 'mt'
adata.var['mt'] = adata.var_names.str.startswith('MT-') 
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
# num of genes expressed in count matrix /  total counts per cell / percentage of mitochondrial genes
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
# plot to know del range
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
# del abnormal genes
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]
# normalize
sc.pp.normalize_total(adata, target_sum=1e4)
# logarithmize
sc.pp.log1p(adata)
# identify highly-variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(adata)
adata.raw = adata
adata = adata[:, adata.var.highly_variable]
# regress out effects of total conts and percentage of m-genes expressed
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
# scale var into unit var
sc.pp.scale(adata, max_value=10)
adata.obs
sc.tl.pca(adata, svd_solver='arpack')
# 2D plot on a map
sc.pl.pca(adata, color='CST3')
sc.pl.pca_variance_ratio(adata)
sc.tl.tsne(adata)
sc.pl.tsne(adata,color=['CST3'],use_raw=False)
from keras.callbacks import Callback
class LossHistory(Callback):
    '''record loss on train and validation process'''
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
 
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
 

    def plot(self):
        iters = range(len(self.losses['epoch']))
        # loss
        plt.plot(iters, self.losses['epoch'], 'g', label='train loss')
        # val_loss
        plt.plot(iters, self.val_loss['epoch'], 'k', label='val loss')
        plt.ylabel('loss')
        plt.xlabel('epoch') 
        plt.legend()
        plt.grid(True)
        plt.show()
        pass
history = LossHistory()
dshape = adata.X.shape
dshape
from keras.layers import Input,Dense,Dropout
from keras.models import Model
act = 'tanh'
inputs = Input(shape=(dshape[1],),name='inputs')
encode1 = Dense(2**10,activation=act,name='encode_layer1')(inputs)
# encode10 = Dropout(0.3)(encode1)
encode2 = Dense(2**6,activation=act,name='encode_layer2')(encode1)
encode3 = Dense(2**4,activation=act,name='encode_layer3')(encode2)

code = Dense(2**1,name='c')(encode3)
decode1 = Dense(2**4,activation=act,name='decode_layer1')(code)
decode2 = Dense(2**6,activation=act,name='decode_layer2')(decode1)
# decode20 = Dropout(0.3)(decode2)
decode3 = Dense(2**10,activation=act,name='decode_layer3')(decode2)
outputs = Dense(dshape[1],activation=act,name='outputs')(decode3)
auto_encoder = Model(inputs=[inputs],outputs=[outputs])
auto_encoder.compile(loss='mse',optimizer='adam')
auto_encoder.fit(adata.X,adata.X,epochs=2**2,batch_size=2**3,callbacks=[history],validation_split=0.1)
history.plot()
encoder = Model(inputs=[inputs],outputs=[code])
Point = encoder.predict(adata.X)
# where's color from ?????? n_genes was choosen
plt.scatter(Point[:,0],Point[:,1],c=adata.obs['n_genes'])