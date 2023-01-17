# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from trackml.dataset import load_event, load_dataset
from trackml.score import score_event
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


path_to_test="../input/train_1/"
event_prefix='event000001000'

hits, cells, particles, truth = load_event(os.path.join(path_to_test, event_prefix))

def get_nn_tuples(all_nns):
    key_value_map = dict()
    for nns in all_nns:
        sz = len(nns)
        if sz > 0:
            if sz in key_value_map.keys():
                key_value_map[sz] = key_value_map[sz] + 1
            else:
                key_value_map[sz] = 1

    print(key_value_map)
    nn_tuples = []
    for nns in all_nns:
        nn_tuples.extend(list(totuple(nns)))

    print("length of nn tuples " + str(len(nn_tuples)))
    return nn_tuples


def filter_z_component(hits, axis, threshold):
    hits_z_component = hits.z
    key_value_map = dict()

    filter_set_max_hits = set()

    # count the no of hits in one z point
    for n in hits_z_component:
        if n in key_value_map.keys():
            key_value_map[n] = key_value_map[n] + 1
        else:
            key_value_map[n] = 1

    sorted_key_value = [(k, key_value_map[k]) for k in sorted(key_value_map, key=key_value_map.get, reverse=True)]

    # looking for disks having maximum hists
    for k, v in sorted_key_value:
        if k > axis and v > threshold:
            #print((k, v))
            filter_set_max_hits.add(k)

    print("size after filter  " + str(len(filter_set_max_hits)))

    return filter_set_max_hits


def get_nn_tuples_list(tree, start, radius):
    all_nn_indices = tree.query_radius(start, r=radius)

    all_nns = [
        [start[idx] for idx in nn_indices if idx != i]
        for i, nn_indices in enumerate(all_nn_indices)
    ]
    return all_nns


def get_nn_tuples_representative(tree, start, radius):
    all_nn_indices = tree.query_radius(start, r=radius)

    all_nns = [
        [start[idx] for idx in nn_indices if idx != i]
        for i, nn_indices in enumerate(all_nn_indices)
    ]

    alls = []
    for nns in all_nns:
        if len(nns) > 0:
            alls.append(nns.pop())

    return alls

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def plot_nn_tuples(nn_tuples):
    dt = np.dtype('float,float,float')
    xarr = np.array(nn_tuples, dtype=dt)
    L = xarr['f0']
    M = xarr['f1']
    N = xarr['f2']

    fig = plt.figure(1)
    plt.suptitle('Plot of major points and their nns  ', fontsize=16)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(L, M, N, zdir='z')

    


def plot_all_data(hits):
    fig = plt.figure(2)
    plt.suptitle('Plot all data points ', fontsize=16)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(hits.x, hits.y, hits.z, zdir='z')

    

plot_all_data(hits)

# look at axis means positive part or negative part in z axis and threshold is number of instances in one z point
filter_set_max_hits = filter_z_component(hits, 0, 900)

hits_stacked_data = np.dstack((hits.x, hits.y, hits.z))

hits_tuples = []
for row in hits_stacked_data:
    hits_tuples.extend(totuple(row))

print(len(hits_tuples))

# filter tuples which are present in points we select in z axis
filtered_hits_tuples = [t for t in hits_tuples if t[2] in filter_set_max_hits]

print(len(filtered_hits_tuples))

# feed data to KDTree
kdtree = KDTree(filtered_hits_tuples, leaf_size=5)

# get nearest neighbor points from filtered_hits_tuples
# all_nns are list of list of nn co ordinates
all_nns = get_nn_tuples_list(kdtree, filtered_hits_tuples, 1.5)

# print distance with no of instance found and return tuple
nn_tuples = get_nn_tuples(all_nns)
plot_nn_tuples(nn_tuples)
plt.show()