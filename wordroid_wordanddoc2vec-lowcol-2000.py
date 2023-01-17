!pip install git+https://github.com/darecophoenixx/wordroid.sblo.jp
%matplotlib inline

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



import os.path

import sys

import re

import itertools

import csv

import datetime

import pickle

import random

from collections import defaultdict, Counter

import gc



import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import seaborn as sns

import pandas as pd

import numpy as np

import scipy

import gensim

from sklearn.metrics import f1_score, classification_report, confusion_matrix, log_loss

from sklearn.model_selection import train_test_split

import gensim

from keras.preprocessing.sequence import skipgrams

import tensorflow as tf
def hexbin(x, y, color, **kwargs):

    cmap = sns.light_palette(color, as_cmap=True)

    plt.hexbin(x, y, cmap=cmap, **kwargs)

def scatter(x, y, color, **kwargs):

    plt.scatter(x, y, marker='.')
NN_word = 2000

NN_sentence = 10000

NN_SEG = 7
product_list = [ee+1 for ee in range(NN_word)]

user_list = [ee+1 for ee in range(NN_sentence)]
a, _ = divmod(len(user_list), NN_SEG)

print(a)

cls_user = [int(user_id / (a+1)) for user_id in range(1, 1+len(user_list))]
a, _ = divmod(len(product_list), NN_SEG)

print(a)

cls_prod = [int(prod_id / (a+1)) for prod_id in range(1, 1+len(product_list))]
random.seed(0)



X_list = []



for ii in range(len(user_list)):

    cls = cls_user[ii]

    product_group = np.array(product_list)[np.array(cls_prod) == cls]

    nword = random.randint(5, 20)

    prods = random.sample(product_group.tolist(), nword)

    irow = np.zeros((1,NN_word))

    irow[0,np.array(prods)-1] = 1

    X_list.append(irow)



X = np.concatenate(X_list)

print(X.shape)

X
X_df = pd.DataFrame(X, dtype=int)

X_df.index = ['r'+ee.astype('str') for ee in (np.arange(X_df.shape[0])+1)]

X_df.columns = ['c'+ee.astype('str') for ee in np.arange(X_df.shape[1])+1]

print(X_df.shape)

X_df.head()
X_df.values.shape
plt.figure(figsize=(10, 10))

plt.imshow(X_df.values.T)
from feature_eng import lowcols
wd2v = lowcols.WD2vec(X_df)

wd2v
num_features = 5



models = wd2v.make_model(num_user=X_df.shape[0], num_product=NN_word, num_features=num_features)

print('\n\n##################### model >>>')

model = models['model']

model.summary()
wgt_user = wd2v.get_wgt_byrow()

print(wgt_user.shape)

df = pd.DataFrame(wgt_user[:,:5])

sns.set_context('paper')

g = sns.PairGrid(df, height=3.5)

g.map_diag(plt.hist, edgecolor="w")

g.map_lower(scatter)

g.map_upper(hexbin)
wgt_lm = wd2v.get_wgt_bycol()

print(wgt_lm.shape)

df = pd.DataFrame(wgt_lm[:,:5])

sns.set_context('paper')

g = sns.PairGrid(df, height=3.5)

g.map_diag(plt.hist, edgecolor="w")

g.map_lower(scatter)

g.map_upper(hexbin)
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau



# def lr_schedule(epoch):

#     def reduce(epoch, lr):

#         if divmod(epoch,4)[1] == 3:

#             lr *= (1/8)

#         elif divmod(epoch,4)[1] == 2:

#             lr *= (1/4)

#         elif divmod(epoch,4)[1] == 1:

#             lr *= (1/2)

#         elif divmod(epoch,4)[1] == 0:

#             pass

#         return lr

    

#     lr0 = 0.01

#     epoch1 = 8

#     epoch2 = 8

#     epoch3 = 8

#     epoch4 = 8

    

#     if epoch<epoch1:

#         lr = lr0

#         #lr = reduce(epoch, lr)

#     elif epoch<epoch1+epoch2:

#         lr = lr0/2

#         #lr = reduce(epoch, lr)

#     elif epoch<epoch1+epoch2+epoch3:

#         lr = lr0/4

#         #lr = reduce(epoch, lr)

#     elif epoch<epoch1+epoch2+epoch3+epoch4:

#         lr = lr0/8

#         #lr = reduce(epoch, lr)

#     else:

#         lr = lr0/16

    

#     print('Learning rate: ', lr)

#     return lr



# lr_scheduler = LearningRateScheduler(lr_schedule)

# callbacks = [lr_scheduler]



# hst = wd2v.train(epochs=32, batch_size=32, verbose=2,

#            callbacks=callbacks)
%%time

hst = wd2v.train(epochs=100, batch_size=32, verbose=2)
hst_history = hst.history
fig, ax = plt.subplots(1, 3, figsize=(20,5))

ax[0].set_title('loss')

ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["loss"], label="Train loss")

ax[1].set_title('acc')

ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["acc"], label="accuracy")

ax[2].set_title('learning rate')

ax[2].plot(list(range(len(hst_history["loss"]))), hst_history["lr"], label="learning rate")

ax[0].legend()

ax[1].legend()

ax[2].legend()
wgt_prod = wd2v.get_wgt_bycol()

print(wgt_prod.shape)

df = pd.DataFrame(wgt_prod[:,:5])

sns.set_context('paper')

g = sns.PairGrid(df, height=3.5)

g.map_diag(plt.hist, edgecolor="w")

g.map_lower(scatter)

g.map_upper(hexbin)
wgt_prod = wd2v.get_wgt_bycol()

print(wgt_prod.shape)

df = pd.DataFrame(wgt_prod[:,:5])

df['cls'] = ['c'+str(ii) for ii in cls_prod]

sns.pairplot(df, markers='o', hue='cls', height=3.5, diag_kind='hist')
wgt_user = wd2v.get_wgt_byrow()

print(wgt_user.shape)

df = pd.DataFrame(wgt_user[:,:5])

sns.set_context('paper')

g = sns.PairGrid(df, height=3.5)

g.map_diag(plt.hist, edgecolor="w")

g.map_lower(scatter)

g.map_upper(hexbin)
wgt_user = wd2v.get_wgt_byrow()

print(wgt_user.shape)

df = pd.DataFrame(wgt_user[:,:5])

df['cls'] = ['c'+str(ii) for ii in cls_user]

sns.pairplot(df, markers='o', hue='cls', height=3.5, diag_kind='hist')
'''show row side and col side at the same time'''

df1 = pd.DataFrame(wgt_prod)

df1['cls'] = ['c'+str(ii) for ii in cls_prod]

df2 = pd.DataFrame(wgt_user)

df2['cls'] = ['r'+str(ii) for ii in cls_user]

df = pd.concat([df2, df1])

df.head()



sns.pairplot(df, markers=['.']*7+['s']*7, hue='cls', height=3.5, diag_kind='hist')