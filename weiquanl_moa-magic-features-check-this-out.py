!pip install --upgrade seaborn
# system

import os, time, datetime

# data structure

import pandas as pd

import numpy as np



# math

from scipy import stats



# model

import tensorflow as tf

from tensorflow import keras

from tensorflow.python.keras.utils.data_utils import Sequence

from sklearn.utils import class_weight

from sklearn.preprocessing import LabelEncoder

from sklearn import manifold, datasets



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.ticker import NullFormatter



# utilities

from collections import OrderedDict

from functools import partial

from time import time

import warnings

warnings.simplefilter("ignore")

sns.__version__
root_dir = '../input/lish-moa/'

os.listdir(root_dir)
train_features_dir = root_dir + 'train_features.csv'

train_targets_dir = root_dir + 'train_targets_scored.csv'

test_features_dir = root_dir + 'test_features.csv'

train_features = pd.read_csv(train_features_dir)

train_targets = pd.read_csv(train_targets_dir).drop(columns = 'sig_id')

test_features = pd.read_csv(test_features_dir)

test_id = test_features['sig_id']
def preprocess(df):

#     df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 1, 'ctl_vehicle': 0})

#     df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

#     df.loc[:, 'cp_time'] = df.loc[:, 'cp_time'].map({24: 0, 48: 1, 72:2})

    del df['sig_id']

    return df

train_features = preprocess(train_features)

test_features = preprocess(test_features)
feature_names = list(train_features.columns)

target_names = list(train_targets.columns)
MoA_sum = train_targets.sum().to_frame().reset_index(drop=False).rename(columns={"index": "MoA", 0: "sum"}).sort_values(ascending = False, by= 'sum')



fig, ax = plt.subplots()

plt.barh(MoA_sum.head(20)['MoA'], MoA_sum.head(20)['sum'])

plt.gca().invert_yaxis()

plt.title('The count of MoAs')

plt.show()

MoA_sum.head(20)
pie_data = train_features[['cp_type', 'cp_time', 'cp_dose']].astype(str)

pie_data.insert(3, 'count', 1)

pie_cp_type = pie_data.groupby(by = ['cp_type']).sum().reset_index().sort_values(by = ['cp_type'])

pie_cp_time = pie_data.groupby(by = ['cp_type', 'cp_time']).sum().reset_index().sort_values(by = ['cp_type', 'cp_time'])

pie_cp_dose = pie_data.groupby(by = ['cp_type', 'cp_time', 'cp_dose']).sum().reset_index().sort_values(by = ['cp_type', 'cp_time', 'cp_dose'])

pie_cp_dose
fig, ax = plt.subplots(figsize = (6,6))

plt.pie(labels = pie_cp_type['cp_type'], x = pie_cp_type['count'], radius = 1.2, labeldistance=0.8, wedgeprops=dict(width=0.3, edgecolor='w'))

plt.pie(labels = pie_cp_time['cp_time'], x = pie_cp_time['count'], radius = 0.9, labeldistance=0.8, wedgeprops=dict(width=0.3, edgecolor='w'))

plt.pie(labels = pie_cp_dose['cp_dose'], x = pie_cp_dose['count'], radius = 0.6, labeldistance=0.8, wedgeprops=dict(width=0.3, edgecolor='w'))

ax.set(title='MoA datapoint counting grouped by `cp_type`, `cp_time`, `cp_dose`')

plt.show()
tag = pd.DataFrame(train_features['cp_type'])



cp_dose_time = train_features['cp_dose'].astype(str) + " " + train_features['cp_time'].astype(str) + "hrs " 

tag['cp_dose_time'] = cp_dose_time

cp_dose_time = list(cp_dose_time.drop_duplicates().sort_values())

feature = train_features.drop(columns = ['cp_type','cp_time', 'cp_dose'])

MoA_count = train_targets.sum(axis = 1)

sns.countplot(MoA_count)

plt.title('datapoint counting of concurrent MoA targets')

plt.show()
MoA_concurrent = train_targets.multiply(MoA_count, axis = 0)

MoA_concurrent = MoA_concurrent.reset_index().melt(id_vars=['index'], value_vars=list(train_targets.columns)).drop(columns = 'index')

MoA_concurrent.columns = ['MoA', 'concurrent_level']

MoA_concurrent = MoA_concurrent[MoA_concurrent['concurrent_level'] >=0]

MoA_concurrent.insert(2, 'count', 1)

MoA_concurrent = MoA_concurrent.groupby(['MoA', 'concurrent_level']).sum('count').reset_index(drop= False)

MoA_concurrent = MoA_concurrent.pivot(index='concurrent_level', columns='MoA', values='count').fillna(0)

MoA_concurrent.astype(bool).sum(axis=1).plot.bar(label = 'count', title = 'number of target variable by concurrent level')

plt.show()
kstest_result = np.ones((len(cp_dose_time)), dtype = float)

kstest_result_m = np.ones((feature.shape[1]), dtype = float)

t0 = time()



for i in np.arange(feature.shape[1]):

    tmp = pd.concat([tag, feature.iloc[:,i]], axis = 1).sort_values('cp_dose_time')

    for j in np.arange(len(cp_dose_time)):

        tmp_dose_time = tmp[tmp['cp_dose_time'] == cp_dose_time[j]]

        tmp_dose_time_trt_cp = np.array(tmp_dose_time[tmp_dose_time['cp_type'] == 'trt_cp'].iloc[:,2])

        tmp_dose_time_ctl_vehicle = np.array(tmp_dose_time[tmp_dose_time['cp_type'] == 'ctl_vehicle'].iloc[:,2])

        kstest_result[j] = stats.ks_2samp(tmp_dose_time_trt_cp, tmp_dose_time_ctl_vehicle).pvalue

    kstest_result_m[i] = kstest_result.max()

#     if i >= 100:

#         break

pval_rank = list(pd.DataFrame(kstest_result_m, columns = ['ks_pval']).reset_index(drop = False).sort_values(by = 'ks_pval')['index'])

print('time spend:', time() - t0)
def facetgrid_two_axes(*args, **kwargs):

    data = kwargs.pop('data')

    dual_axis = kwargs.pop('dual_axis')



    ax = plt.gca()

    ax.set_ylabel('Count of MoA')

    

    sns.scatterplot(tmp.iloc[:,2],tmp.iloc[:,3],

                    alpha=0.01)



    ax.set_ylabel('MoA Count')



    ax2 = ax.twinx()

    ax2.set_ylabel('freq')

    sns.kdeplot(data.iloc[:,2], fill = True, hue = data.iloc[:,0], 

                common_norm=False,  legend=False,

                palette=['dodgerblue','coral'],

                alpha=0.5)   

    

for i in pval_rank:

    tmp = pd.concat([tag, feature.iloc[:,i], MoA_count], axis = 1).sort_values('cp_dose_time')

    g = sns.FacetGrid(tmp, col='cp_dose_time') 

    g.map_dataframe(facetgrid_two_axes, dual_axis=True)

    g.fig.suptitle('Distribution of Var: ['+ feature.iloc[:,i].name + "] (blue: trt_cp, red: ctl_vehicle) with MoA targets counting", y= 1.1)

    plt.show()

#     if i >= 100:

#         break